"""Train a lightweight residual CNN on the matched IGS GIM dataset.

Protocol follows the initial experiment requested for this project:

* the final N natural UTC days are a strictly later test set;
* all earlier epochs are shuffled with a fixed seed and split train/validation;
* the CNN predicts ``Final - RT`` and the corrected map is ``RT + prediction``;
* longitude is treated as periodic and the duplicated +180 degree column is
  removed during training, then restored in exported predictions.

The paper-inspired defaults are Adam, lr=5e-4, batch size 64, L1 loss, no
dropout and no weight decay. The checkpoint with minimum validation L1 is used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset


@dataclass(frozen=True)
class Normalization:
    rt_mean: float
    rt_std: float
    residual_mean: float
    residual_std: float


class GIMDataset(TorchDataset):
    def __init__(
        self,
        rt: np.ndarray,
        residual: np.ndarray,
        indices: np.ndarray,
        normalization: Normalization,
    ) -> None:
        self.rt = rt
        self.residual = residual
        self.indices = np.asarray(indices, dtype=np.int64)
        self.norm = normalization

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        source_index = int(self.indices[item])
        x = (self.rt[source_index] - self.norm.rt_mean) / self.norm.rt_std
        y = (
            self.residual[source_index] - self.norm.residual_mean
        ) / self.norm.residual_std
        return (
            torch.from_numpy(np.ascontiguousarray(x[None, :, :])),
            torch.from_numpy(np.ascontiguousarray(y[None, :, :])),
            source_index,
        )


def global_pad(x: torch.Tensor, amount: int = 1) -> torch.Tensor:
    """Circular longitude padding plus replicated polar-edge padding."""
    x = F.pad(x, (amount, amount, 0, 0), mode="circular")
    return F.pad(x, (0, 0, amount, amount), mode="replicate")


class GlobalConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(global_pad(x, self.pad))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = GlobalConv2d(channels, channels)
        self.conv2 = GlobalConv2d(channels, channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.activation(self.conv1(x))
        update = self.conv2(update)
        return self.activation(x + update)


class ResidualCNN(nn.Module):
    """Small same-resolution CNN that predicts a normalized residual map."""

    def __init__(self, channels: int = 32, blocks: int = 4) -> None:
        super().__init__()
        self.head = GlobalConv2d(1, channels)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.tail = GlobalConv2d(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.head(x), inplace=True)
        return self.tail(self.blocks(features))


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def to_python_datetime(value: object) -> datetime:
    return datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
    )


def load_matched_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[datetime], str, str]:
    with Dataset(path) as nc:
        required = {"time", "lat", "lon", "rt_vtec", "residual"}
        missing = required - set(nc.variables)
        if missing:
            raise ValueError(f"Dataset is missing variables: {sorted(missing)}")
        time_var = nc.variables["time"]
        raw_times = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        times = [to_python_datetime(value) for value in raw_times]
        time_values = np.asarray(time_var[:], dtype=np.float64)
        time_units = str(time_var.units)
        time_calendar = str(getattr(time_var, "calendar", "standard"))
        lat = np.asarray(nc.variables["lat"][:], dtype=np.float32)
        lon_full = np.asarray(nc.variables["lon"][:], dtype=np.float32)
        rt_full = np.ma.filled(nc.variables["rt_vtec"][:], np.nan).astype(np.float32)
        residual_full = np.ma.filled(nc.variables["residual"][:], np.nan).astype(np.float32)

    if not np.all(np.diff(time_values) > 0):
        raise ValueError("Time coordinate is not strictly increasing")
    if not np.isclose(lon_full[-1] - lon_full[0], 360.0, atol=1e-5):
        raise ValueError("Expected a global longitude grid with duplicated seam")
    if not np.allclose(rt_full[:, :, 0], rt_full[:, :, -1], atol=1e-6):
        raise ValueError("RT longitude seam is inconsistent")
    if not np.allclose(residual_full[:, :, 0], residual_full[:, :, -1], atol=1e-6):
        raise ValueError("Residual longitude seam is inconsistent")
    if not np.isfinite(rt_full).all() or not np.isfinite(residual_full).all():
        raise ValueError("Training arrays contain NaN or infinite values")

    # The +180 column duplicates -180. Use 72 unique longitudes so circular
    # padding connects +175 directly to -180 without inserting a duplicate.
    return (
        rt_full[:, :, :-1],
        residual_full[:, :, :-1],
        lat,
        lon_full[:-1],
        time_values,
        times,
        time_units,
        time_calendar,
    )


def make_splits(
    times: list[datetime],
    test_days: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    if test_days < 1:
        raise ValueError("test_days must be at least 1")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    last = max(times)
    last_day = datetime(last.year, last.month, last.day)
    test_start = last_day - timedelta(days=test_days - 1)
    test_indices = np.asarray(
        [index for index, value in enumerate(times) if value >= test_start],
        dtype=np.int64,
    )
    earlier = np.asarray(
        [index for index, value in enumerate(times) if value < test_start],
        dtype=np.int64,
    )
    if test_indices.size == 0 or earlier.size < 2:
        raise ValueError("Insufficient samples for the requested split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(earlier)
    validation_count = max(1, int(round(shuffled.size * val_fraction)))
    validation_indices = np.sort(shuffled[:validation_count])
    train_indices = np.sort(shuffled[validation_count:])
    return train_indices, validation_indices, np.sort(test_indices), test_start


def calculate_normalization(
    rt: np.ndarray,
    residual: np.ndarray,
    train_indices: np.ndarray,
) -> Normalization:
    train_rt = rt[train_indices].astype(np.float64, copy=False)
    train_residual = residual[train_indices].astype(np.float64, copy=False)
    normalization = Normalization(
        rt_mean=float(np.mean(train_rt)),
        rt_std=float(np.std(train_rt)),
        residual_mean=float(np.mean(train_residual)),
        residual_std=float(np.std(train_residual)),
    )
    if normalization.rt_std <= 0 or normalization.residual_std <= 0:
        raise ValueError("Training normalization standard deviation is not positive")
    return normalization


def cap_split(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def make_loader(
    dataset: GIMDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    use_amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for inputs, targets, _indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_count = int(inputs.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_samples += batch_count
    return total_loss / total_samples


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalization: Normalization,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _targets, source_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                normalized = model(inputs)
            physical = (
                normalized.float().cpu().numpy()[:, 0]
                * normalization.residual_std
                + normalization.residual_mean
            )
            predictions.append(physical.astype(np.float32))
            indices.append(source_indices.numpy().astype(np.int64))
    all_predictions = np.concatenate(predictions)
    all_indices = np.concatenate(indices)
    order = np.argsort(all_indices)
    return all_indices[order], all_predictions[order]


def metric_row(
    estimate: np.ndarray,
    final: np.ndarray,
    region: str,
    product: str,
) -> dict[str, object]:
    error_final_minus_estimate = final.astype(np.float64) - estimate.astype(np.float64)
    estimate64 = estimate.astype(np.float64)
    final64 = final.astype(np.float64)
    covariance = np.mean(
        (estimate64 - np.mean(estimate64)) * (final64 - np.mean(final64))
    )
    denominator = np.std(estimate64) * np.std(final64)
    return {
        "region": region,
        "product": product,
        "count": int(error_final_minus_estimate.size),
        "bias_final_minus_estimate_tecu": float(np.mean(error_final_minus_estimate)),
        "mae_tecu": float(np.mean(np.abs(error_final_minus_estimate))),
        "rmse_tecu": float(np.sqrt(np.mean(error_final_minus_estimate**2))),
        "correlation": float(covariance / denominator) if denominator > 0 else math.nan,
    }


def evaluate_metrics(
    rt: np.ndarray,
    true_residual: np.ndarray,
    predicted_residual: np.ndarray,
    lat: np.ndarray,
) -> list[dict[str, object]]:
    final = rt + true_residual
    corrected = rt + predicted_residual
    masks = {
        "global": np.ones(lat.size, dtype=bool),
        "low_latitude_abs_lt_30": np.abs(lat) < 30,
        "mid_latitude_abs_30_to_60": (np.abs(lat) >= 30) & (np.abs(lat) < 60),
        "high_latitude_abs_ge_60": np.abs(lat) >= 60,
    }
    rows: list[dict[str, object]] = []
    for region, mask in masks.items():
        rows.append(metric_row(rt[:, mask], final[:, mask], region, "raw_rt"))
        rows.append(metric_row(corrected[:, mask], final[:, mask], region, "cnn_corrected"))
    by_key = {(str(row["region"]), str(row["product"])): row for row in rows}
    for region in masks:
        raw = float(by_key[(region, "raw_rt")]["rmse_tecu"])
        corrected_rmse = float(by_key[(region, "cnn_corrected")]["rmse_tecu"])
        improvement = 100.0 * (raw - corrected_rmse) / raw
        by_key[(region, "cnn_corrected")]["rmse_improvement_percent"] = improvement
        by_key[(region, "raw_rt")]["rmse_improvement_percent"] = 0.0
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_split_csv(
    path: Path,
    times: list[datetime],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    labels = np.full(len(times), "", dtype=object)
    labels[train_indices] = "train"
    labels[validation_indices] = "validation"
    labels[test_indices] = "test"
    rows = [
        {
            "dataset_index": index,
            "time_utc": times[index].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "doy": times[index].timetuple().tm_yday,
            "split": labels[index],
        }
        for index in range(len(times))
    ]
    write_csv(path, rows, ["dataset_index", "time_utc", "doy", "split"])


def save_predictions_netcdf(
    path: Path,
    source_indices: np.ndarray,
    time_values: np.ndarray,
    time_units: str,
    time_calendar: str,
    lat: np.ndarray,
    lon_unique: np.ndarray,
    rt: np.ndarray,
    true_residual: np.ndarray,
    predicted_residual: np.ndarray,
) -> None:
    lon = np.concatenate([lon_unique, [lon_unique[0] + 360.0]]).astype(np.float32)

    def add_seam(array: np.ndarray) -> np.ndarray:
        return np.concatenate([array, array[:, :, :1]], axis=2).astype(np.float32)

    rt_full = add_seam(rt)
    true_full = add_seam(true_residual)
    predicted_full = add_seam(predicted_residual)
    final_full = rt_full + true_full
    corrected_full = rt_full + predicted_full
    temporary = path.with_suffix(path.suffix + ".tmp")
    with Dataset(temporary, "w", format="NETCDF4") as nc:
        nc.createDimension("time", source_indices.size)
        nc.createDimension("lat", lat.size)
        nc.createDimension("lon", lon.size)
        time_var = nc.createVariable("time", "f8", ("time",))
        lat_var = nc.createVariable("lat", "f4", ("lat",))
        lon_var = nc.createVariable("lon", "f4", ("lon",))
        index_var = nc.createVariable("source_dataset_index", "i4", ("time",))
        time_var.units = time_units
        time_var.calendar = time_calendar
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        time_var[:] = time_values[source_indices]
        lat_var[:] = lat
        lon_var[:] = lon
        index_var[:] = source_indices
        chunks = (1, lat.size, lon.size)
        for name, array, long_name in (
            ("rt_vtec", rt_full, "original real-time VTEC"),
            ("final_vtec", final_full, "matched Final VTEC"),
            ("true_residual", true_full, "final_vtec - rt_vtec"),
            ("predicted_residual", predicted_full, "CNN-predicted residual"),
            ("corrected_vtec", corrected_full, "rt_vtec + predicted_residual"),
        ):
            variable = nc.createVariable(
                name,
                "f4",
                ("time", "lat", "lon"),
                zlib=True,
                complevel=4,
                shuffle=True,
                chunksizes=chunks,
            )
            variable.units = "TECU"
            variable.long_name = long_name
            variable[:] = array
        nc.title = "Residual CNN predictions for the final three-day GIM test set"
        nc.residual_definition = "Final - RT"
    os.replace(temporary, path)


def save_plots(
    output_dir: Path,
    history: list[dict[str, float]],
    times: list[datetime],
    test_indices: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    rt: np.ndarray,
    true_residual: np.ndarray,
    predicted_residual: np.ndarray,
) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.plot(epochs, [row["train_l1_normalized"] for row in history], label="Train")
    ax.plot(epochs, [row["validation_l1_normalized"] for row in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Normalized L1 loss", title="Training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(fig)

    final = rt + true_residual
    corrected = rt + predicted_residual
    raw_rmse = np.sqrt(np.mean(true_residual.astype(np.float64) ** 2, axis=(1, 2)))
    corrected_error = true_residual.astype(np.float64) - predicted_residual.astype(np.float64)
    corrected_rmse = np.sqrt(np.mean(corrected_error**2, axis=(1, 2)))
    test_times = [times[index] for index in test_indices]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(test_times, raw_rmse, label="Raw RT", linewidth=1.2)
    ax.plot(test_times, corrected_rmse, label="CNN corrected", linewidth=1.2)
    ax.set(xlabel="UTC", ylabel="RMSE (TECU)", title="Test RMSE by epoch")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "test_rmse_by_time.png", dpi=160)
    plt.close(fig)

    positions = np.unique(np.linspace(0, len(test_indices) - 1, min(6, len(test_indices))).astype(int))
    fig, axes = plt.subplots(len(positions), 5, figsize=(16, 2.5 * len(positions)), constrained_layout=True)
    if len(positions) == 1:
        axes = np.asarray([axes])
    for row, position in enumerate(positions):
        stamp = test_times[position].strftime("%Y-%m-%d %H:%M")
        maps = (
            (rt[position], "RT", "viridis", False),
            (final[position], "Final", "viridis", False),
            (corrected[position], "Corrected", "viridis", False),
            (true_residual[position], "True residual", "RdBu_r", True),
            (predicted_residual[position], "Predicted residual", "RdBu_r", True),
        )
        vtec_min = float(np.min([rt[position].min(), final[position].min(), corrected[position].min()]))
        vtec_max = float(np.max([rt[position].max(), final[position].max(), corrected[position].max()]))
        residual_limit = float(
            np.percentile(
                np.abs(np.concatenate([true_residual[position].ravel(), predicted_residual[position].ravel()])),
                99,
            )
        )
        for column, (data, label, cmap, symmetric) in enumerate(maps):
            axis = axes[row, column]
            if symmetric:
                axis.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=-residual_limit, vmax=residual_limit)
            else:
                axis.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vtec_min, vmax=vtec_max)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{stamp} - {label}", fontsize=8)
    fig.savefig(output_dir / "test_examples.png", dpi=140)
    plt.close(fig)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/share/home/u23114/tj23114/data/Yaoyaping_data/GIM2021/IGS/"
            "GIM_CNN_dataset_2021_DOY091_181.nc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN/"
            "results/GIM_CNN_baseline_seed42"
        ),
    )
    parser.add_argument("--test-days", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_amp = bool(args.amp and device.type == "cuda")

    print(f"Device: {device}", flush=True)
    print("Loading matched NetCDF into memory ...", flush=True)
    (
        rt,
        residual,
        lat,
        lon,
        time_values,
        times,
        time_units,
        time_calendar,
    ) = load_matched_dataset(args.dataset.resolve())
    train_indices, validation_indices, test_indices, test_start = make_splits(
        times, args.test_days, args.val_fraction, args.seed
    )
    full_split_counts = {
        "train": int(train_indices.size),
        "validation": int(validation_indices.size),
        "test": int(test_indices.size),
    }

    epochs = args.epochs
    patience = args.patience
    channels = args.channels
    blocks = args.blocks
    batch_size = args.batch_size
    num_workers = args.num_workers
    if args.smoke_test:
        train_indices = cap_split(train_indices, 96, args.seed)
        validation_indices = cap_split(validation_indices, 32, args.seed + 1)
        test_indices = cap_split(test_indices, 32, args.seed + 2)
        epochs = 1
        patience = 1
        channels = min(channels, 8)
        blocks = min(blocks, 1)
        batch_size = min(batch_size, 16)
        num_workers = 0

    normalization = calculate_normalization(rt, residual, train_indices)
    save_split_csv(
        output_dir / "data_split.csv",
        times,
        train_indices,
        validation_indices,
        test_indices,
    )
    print(
        f"Split: train={train_indices.size}, validation={validation_indices.size}, "
        f"test={test_indices.size}; test starts {test_start:%Y-%m-%d %H:%M} UTC",
        flush=True,
    )

    train_loader = make_loader(
        GIMDataset(rt, residual, train_indices, normalization),
        batch_size,
        True,
        num_workers,
        device,
    )
    validation_loader = make_loader(
        GIMDataset(rt, residual, validation_indices, normalization),
        batch_size,
        False,
        num_workers,
        device,
    )
    test_loader = make_loader(
        GIMDataset(rt, residual, test_indices, normalization),
        batch_size,
        False,
        num_workers,
        device,
    )

    model = ResidualCNN(channels=channels, blocks=blocks).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    loss_function = nn.L1Loss()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch versions before torch.amp GradScaler
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Model parameters: {parameter_count:,}", flush=True)

    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best_model.pt"
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            scaler,
            use_amp,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            None,
            scaler,
            use_amp,
        )
        seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": float(epoch),
                "train_l1_normalized": train_loss,
                "validation_l1_normalized": validation_loss,
                "seconds": seconds,
            }
        )
        improved = validation_loss < best_loss - args.min_delta
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": {"channels": channels, "blocks": blocks},
                    "normalization": asdict(normalization),
                    "best_epoch": best_epoch,
                    "best_validation_l1_normalized": best_loss,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch:03d}: train={train_loss:.6f} "
            f"val={validation_loss:.6f} seconds={seconds:.1f} "
            f"best={best_loss:.6f}@{best_epoch}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    source_indices, predicted_residual = predict(
        model, test_loader, device, normalization, use_amp
    )
    if not np.array_equal(source_indices, np.sort(test_indices)):
        raise RuntimeError("Test prediction indices do not match the requested split")
    test_indices = source_indices
    test_rt = rt[test_indices]
    test_true_residual = residual[test_indices]
    rows = evaluate_metrics(
        test_rt, test_true_residual, predicted_residual, lat
    )
    write_csv(
        output_dir / "test_metrics.csv",
        rows,
        [
            "region",
            "product",
            "count",
            "bias_final_minus_estimate_tecu",
            "mae_tecu",
            "rmse_tecu",
            "correlation",
            "rmse_improvement_percent",
        ],
    )
    write_csv(
        output_dir / "training_history.csv",
        history,
        ["epoch", "train_l1_normalized", "validation_l1_normalized", "seconds"],
    )
    save_predictions_netcdf(
        output_dir / "test_predictions.nc",
        test_indices,
        time_values,
        time_units,
        time_calendar,
        lat,
        lon,
        test_rt,
        test_true_residual,
        predicted_residual,
    )
    save_plots(
        output_dir,
        history,
        times,
        test_indices,
        lat,
        lon,
        test_rt,
        test_true_residual,
        predicted_residual,
    )

    global_rows = {str(row["product"]): row for row in rows if row["region"] == "global"}
    summary = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "model": "lightweight residual CNN",
        "model_parameters": parameter_count,
        "channels": channels,
        "residual_blocks": blocks,
        "loss": "L1 on normalized residual",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": batch_size,
        "seed": args.seed,
        "test_days": args.test_days,
        "test_start_utc": test_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_full_split_counts": full_split_counts,
        "actual_run_split_counts": {
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "normalization_from_training_only": asdict(normalization),
        "best_epoch": best_epoch,
        "best_validation_l1_normalized": best_loss,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - training_start,
        "test_global_raw_rt": global_rows["raw_rt"],
        "test_global_cnn_corrected": global_rows["cnn_corrected"],
        "smoke_test": bool(args.smoke_test),
    }
    temporary_summary = output_dir / "run_summary.json.tmp"
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_summary, output_dir / "run_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
