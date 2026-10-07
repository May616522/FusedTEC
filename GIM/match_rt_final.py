"""Match 20-minute IGS RT-GIMs to rotated/interpolated IGS Final GIMs.

The temporal interpolation follows Eq. (1) in Iten et al. (2025): both
bracketing final TEC maps are rotated to the requested epoch before their
time-weighted combination. All timestamps are UTC and matching is driven by
the real-time epochs that actually exist.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, date2num
import numpy as np

from ionex_reader import IonexMap, read_first_tec_epoch, read_ionex_tec_maps


UTC = timezone.utc


@dataclass
class RunningStats:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sum_diff2: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        xv = x[mask].astype(np.float64, copy=False)
        yv = y[mask].astype(np.float64, copy=False)
        diff = yv - xv
        self.count += int(xv.size)
        self.sum_x += float(np.sum(xv))
        self.sum_y += float(np.sum(yv))
        self.sum_x2 += float(np.dot(xv, xv))
        self.sum_y2 += float(np.dot(yv, yv))
        self.sum_xy += float(np.dot(xv, yv))
        self.sum_diff += float(np.sum(diff))
        self.sum_abs_diff += float(np.sum(np.abs(diff)))
        self.sum_diff2 += float(np.dot(diff, diff))

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "bias_tecu": math.nan,
                "mae_tecu": math.nan,
                "rmse_tecu": math.nan,
                "std_tecu": math.nan,
                "correlation": math.nan,
            }
        n = float(self.count)
        bias = self.sum_diff / n
        variance = max(self.sum_diff2 / n - bias * bias, 0.0)
        covariance = self.sum_xy - self.sum_x * self.sum_y / n
        var_x = self.sum_x2 - self.sum_x * self.sum_x / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        correlation = (
            covariance / math.sqrt(var_x * var_y)
            if var_x > 0.0 and var_y > 0.0
            else math.nan
        )
        return {
            "count": self.count,
            "bias_tecu": bias,
            "mae_tecu": self.sum_abs_diff / n,
            "rmse_tecu": math.sqrt(self.sum_diff2 / n),
            "std_tecu": math.sqrt(variance),
            "correlation": correlation,
        }


def doy_start(year: int, doy: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1)


def iso_time(value: datetime | None) -> str:
    return "" if value is None else value.strftime("%Y-%m-%dT%H:%M:%SZ")


def discover_files(root: Path) -> list[Path]:
    suffixes = (".z", ".gz", ".i", ".inx")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith(suffixes)
    )


def inventory_rt(
    root: Path,
    start: datetime,
    end_exclusive: datetime,
) -> tuple[dict[datetime, Path], list[dict[str, object]], list[dict[str, str]]]:
    by_time: dict[datetime, Path] = {}
    inventory: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in discover_files(root):
        try:
            epoch = read_first_tec_epoch(path)
        except Exception as exc:  # keep the full run auditable
            errors.append({"stage": "rt_inventory", "filepath": str(path), "error": str(exc)})
            continue
        in_range = start <= epoch < end_exclusive
        duplicate = epoch in by_time
        if in_range and not duplicate:
            by_time[epoch] = path
        elif duplicate:
            errors.append(
                {
                    "stage": "rt_inventory",
                    "filepath": str(path),
                    "error": f"duplicate epoch; kept {by_time[epoch]}",
                }
            )
        inventory.append(
            {
                "time": iso_time(epoch),
                "year": epoch.year,
                "doy": epoch.timetuple().tm_yday,
                "filename": path.name,
                "filepath": str(path),
                "valid": int(in_range and not duplicate),
            }
        )
    inventory.sort(key=lambda row: (str(row["time"]), str(row["filepath"])))
    return by_time, inventory, errors


def load_final_series(
    root: Path,
    start: datetime,
    end_inclusive: datetime,
) -> tuple[list[datetime], np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    maps_by_time: dict[datetime, IonexMap] = {}
    errors: list[dict[str, str]] = []
    start_doy = start.timetuple().tm_yday
    end_doy = end_inclusive.timetuple().tm_yday
    input_paths = discover_files(root)
    if start.year == end_inclusive.year:
        # The supplied archive is organized as one three-digit DOY directory
        # per daily IONEX file. Restricting it here avoids decompressing the
        # rest of the year. The final DOY is included for its 00:00 endpoint.
        input_paths = [
            path
            for path in input_paths
            if not path.parent.name.isdigit()
            or start_doy <= int(path.parent.name) <= end_doy
        ]
    for path in input_paths:
        try:
            maps = read_ionex_tec_maps(path)
        except Exception as exc:
            errors.append({"stage": "final_parse", "filepath": str(path), "error": str(exc)})
            continue
        relevant = [item for item in maps if start <= item.time <= end_inclusive]
        for item in relevant:
            previous = maps_by_time.get(item.time)
            if previous is not None:
                if (
                    not np.allclose(previous.lat, item.lat)
                    or not np.allclose(previous.lon, item.lon)
                ):
                    raise ValueError(f"Final grid changes at duplicate epoch {item.time}")
                difference = np.nanmax(np.abs(previous.vtec - item.vtec))
                if difference > 1e-5:
                    errors.append(
                        {
                            "stage": "final_boundary_duplicate",
                            "filepath": str(path),
                            "error": (
                                f"duplicate {iso_time(item.time)} differs by "
                                f"up to {difference:.6g} TECU; selected later "
                                "daily file's 00:00 map"
                            ),
                        }
                    )
                # Daily IONEX files contain the next midnight as their last
                # map, while the next file repeats it as its first. Select the
                # map from the day to which 00:00 belongs. This also guarantees
                # that DOY 182 supplies the right endpoint requested for DOY 181.
            maps_by_time[item.time] = item

    times = sorted(maps_by_time)
    if not times:
        raise RuntimeError("No Final TEC maps were loaded for the requested interval")
    first = maps_by_time[times[0]]
    arrays: list[np.ndarray] = []
    for epoch in times:
        item = maps_by_time[epoch]
        if not np.allclose(first.lat, item.lat) or not np.allclose(first.lon, item.lon):
            raise ValueError(f"Final grid changes at {iso_time(epoch)}")
        arrays.append(item.vtec)
    return times, np.stack(arrays), first.lat, first.lon, errors


def rotate_global_map(
    data: np.ndarray,
    lon: np.ndarray,
    offset_degrees: float,
) -> np.ndarray:
    """Evaluate E(latitude, longitude + offset) on a periodic regular grid."""
    if data.shape[1] != lon.size:
        raise ValueError("Longitude coordinate and data width differ")
    if not np.isclose(lon[-1] - lon[0], 360.0, atol=1e-5):
        raise ValueError("Longitude grid must include a duplicated global seam")
    unique = data[:, :-1]
    step = float(lon[1] - lon[0])
    shift = offset_degrees / step
    base = math.floor(shift)
    fraction = shift - base
    indices = np.arange(unique.shape[1])
    left = unique[:, (indices + base) % unique.shape[1]]
    right = unique[:, (indices + base + 1) % unique.shape[1]]
    rotated_unique = (1.0 - fraction) * left + fraction * right
    return np.concatenate([rotated_unique, rotated_unique[:, :1]], axis=1).astype(
        np.float32, copy=False
    )


def final_bracket(
    target: datetime,
    final_times: list[datetime],
    max_gap: timedelta,
) -> tuple[int, int, float] | None:
    position = bisect_left(final_times, target)
    if position < len(final_times) and final_times[position] == target:
        return position, position, 0.0
    if position == 0 or position == len(final_times):
        return None
    before = position - 1
    after = position
    interval = final_times[after] - final_times[before]
    if interval <= timedelta(0) or interval > max_gap:
        return None
    weight_after = (target - final_times[before]).total_seconds() / interval.total_seconds()
    return before, after, weight_after


def interpolate_rotated_final(
    target: datetime,
    bracket: tuple[int, int, float],
    final_times: list[datetime],
    final_maps: np.ndarray,
    lon: np.ndarray,
    rotation_rate: float,
) -> np.ndarray:
    before, after, weight_after = bracket
    if before == after:
        return final_maps[before].astype(np.float32, copy=True)
    hours_before = (target - final_times[before]).total_seconds() / 3600.0
    hours_after = (target - final_times[after]).total_seconds() / 3600.0
    map_before = rotate_global_map(
        final_maps[before], lon, rotation_rate * hours_before
    )
    map_after = rotate_global_map(final_maps[after], lon, rotation_rate * hours_after)
    return (
        (1.0 - weight_after) * map_before + weight_after * map_after
    ).astype(np.float32, copy=False)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def add_map_plot(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    label: str,
    path: Path,
    symmetric: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    kwargs: dict[str, object] = {}
    if symmetric:
        limit = float(np.nanpercentile(np.abs(data), 99))
        kwargs.update(vmin=-limit, vmax=limit, cmap="RdBu_r")
    image = ax.imshow(
        data,
        origin="lower",
        extent=[float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])],
        aspect="auto",
        **kwargs,
    )
    ax.set(title=title, xlabel="Longitude (deg)", ylabel="Latitude (deg)")
    fig.colorbar(image, ax=ax, label=label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_qc(
    output_dir: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    example: tuple[datetime, np.ndarray, np.ndarray, np.ndarray],
    samples: list[tuple[datetime, np.ndarray, np.ndarray, np.ndarray]],
    mean_residual: np.ndarray,
    histogram_edges: np.ndarray,
    histogram_counts: np.ndarray,
    histogram_outside: int,
    metric_times: list[datetime],
    bias_by_time: list[float],
    rmse_by_time: list[float],
    rmse_by_latitude: np.ndarray,
) -> None:
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    example_time, rt, final, residual = example
    stamp = example_time.strftime("%Y-%m-%d %H:%M UTC")
    add_map_plot(rt, lat, lon, f"RT-GIM: {stamp}", "VTEC (TECU)", qc_dir / "01_rt_example.png")
    add_map_plot(
        final,
        lat,
        lon,
        f"Rotated/interpolated Final GIM: {stamp}",
        "VTEC (TECU)",
        qc_dir / "02_final_example.png",
    )
    add_map_plot(
        residual,
        lat,
        lon,
        f"Final - RT residual: {stamp}",
        "Residual (TECU)",
        qc_dir / "03_residual_example.png",
        symmetric=True,
    )
    add_map_plot(
        mean_residual,
        lat,
        lon,
        "Mean residual over all matched epochs",
        "Mean residual (TECU)",
        qc_dir / "04_global_mean_residual.png",
        symmetric=True,
    )

    centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.step(centers, histogram_counts, where="mid")
    ax.set(
        title=f"Residual distribution (outside plotted range: {histogram_outside:,})",
        xlabel="Final - RT (TECU)",
        ylabel="Grid-cell count",
        yscale="log",
    )
    ax.grid(alpha=0.25)
    fig.savefig(qc_dir / "05_residual_histogram.png", dpi=160)
    plt.close(fig)

    for filename, values, ylabel, title in (
        ("06_rmse_by_time.png", rmse_by_time, "RMSE (TECU)", "Global RMSE by epoch"),
        ("07_bias_by_time.png", bias_by_time, "Bias (TECU)", "Global bias by epoch"),
    ):
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        ax.plot(metric_times, values, linewidth=0.55)
        ax.set(title=title, xlabel="UTC", ylabel=ylabel)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(alpha=0.25)
        fig.savefig(qc_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot(rmse_by_latitude, lat)
    ax.set(
        title="Residual RMSE by latitude",
        xlabel="RMSE (TECU)",
        ylabel="Latitude (deg)",
    )
    ax.grid(alpha=0.25)
    fig.savefig(qc_dir / "08_rmse_by_latitude.png", dpi=160)
    plt.close(fig)

    if samples:
        fig, axes = plt.subplots(
            len(samples), 3, figsize=(12, 2.25 * len(samples)), constrained_layout=True
        )
        if len(samples) == 1:
            axes = np.asarray([axes])
        for row, (epoch, rt_map, final_map, residual_map) in enumerate(samples):
            for column, (data, title, cmap) in enumerate(
                ((rt_map, "RT", "viridis"), (final_map, "Final", "viridis"), (residual_map, "Residual", "RdBu_r"))
            ):
                axis = axes[row, column]
                if column == 2:
                    limit = float(np.nanpercentile(np.abs(data), 99))
                    axis.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
                else:
                    axis.imshow(data, origin="lower", aspect="auto", cmap=cmap)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(f"{epoch:%Y-%m-%d %H:%M} - {title}", fontsize=8)
        fig.savefig(qc_dir / "09_random_10_epochs.png", dpi=140)
        plt.close(fig)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--start-doy", type=int, default=91)
    parser.add_argument("--end-doy", type=int, default=181)
    parser.add_argument("--excluded-doy", type=int, nargs="*", default=[138, 139])
    parser.add_argument(
        "--rt-root", type=Path, default=Path(r"F:\FusedTec\Data\GIM2021\IGS\RT")
    )
    parser.add_argument(
        "--final-root", type=Path, default=Path(r"F:\FusedTec\Data\GIM2021\IGS\Final")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(r"F:\FusedTec\Data\GIM2021\IGS\match")
    )
    parser.add_argument("--rotation-rate", type=float, default=15.0)
    parser.add_argument("--max-final-gap-hours", type=float, default=2.01)
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    start = doy_start(args.year, args.start_doy)
    end_exclusive = doy_start(args.year, args.end_doy + 1)
    final_end = end_exclusive
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"GIM_CNN_dataset_{args.year}_DOY{args.start_doy:03d}_{args.end_doy:03d}.nc"
    temporary_dataset = dataset_path.with_suffix(".nc.tmp")
    if dataset_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {dataset_path}")
    if temporary_dataset.exists():
        temporary_dataset.unlink()

    print("[1/6] Loading continuous Final-GIM series ...", flush=True)
    final_times, final_maps, lat, lon, errors = load_final_series(
        args.final_root.resolve(), start, final_end
    )
    print(
        f"      {len(final_times)} unique epochs: {iso_time(final_times[0])} .. "
        f"{iso_time(final_times[-1])}; grid={lat.size}x{lon.size}",
        flush=True,
    )

    print("[2/6] Building RT inventory from IONEX epochs ...", flush=True)
    rt_by_time, inventory, inventory_errors = inventory_rt(
        args.rt_root.resolve(), start, end_exclusive
    )
    errors.extend(inventory_errors)
    write_csv(
        output_dir / "rt_inventory.csv",
        inventory,
        ["time", "year", "doy", "filename", "filepath", "valid"],
    )
    print(f"      {len(rt_by_time)} unique in-range RT epochs", flush=True)

    excluded = set(args.excluded_doy)
    max_gap = timedelta(hours=args.max_final_gap_hours)
    report: list[dict[str, object]] = []
    expected = start
    while expected < end_exclusive:
        doy = expected.timetuple().tm_yday
        rt_path = rt_by_time.get(expected)
        bracket = final_bracket(expected, final_times, max_gap)
        if bracket is None:
            before_time = None
            after_time = None
            weight = math.nan
        else:
            before_time = final_times[bracket[0]]
            after_time = final_times[bracket[1]]
            weight = bracket[2]

        if doy in excluded:
            reason = "excluded_doy"
        elif rt_path is None:
            reason = "missing_rt"
        elif bracket is None:
            reason = "missing_or_nonconsecutive_final_bracket"
        else:
            reason = "ok"
        report.append(
            {
                "time": iso_time(expected),
                "year": expected.year,
                "doy": doy,
                "rt_exists": int(rt_path is not None),
                "rt_filepath": "" if rt_path is None else str(rt_path),
                "final_before_time": iso_time(before_time),
                "final_after_time": iso_time(after_time),
                "final_weight_after": "" if not math.isfinite(weight) else f"{weight:.9f}",
                "final_interp_valid": int(bracket is not None),
                "pair_valid": int(reason == "ok"),
                "dataset_index": "",
                "reason": reason,
            }
        )
        expected += timedelta(minutes=20)

    candidate_rows = [row for row in report if row["reason"] == "ok"]
    sample_targets = set(
        int(value)
        for value in np.linspace(0, max(len(candidate_rows) - 1, 0), min(10, len(candidate_rows)))
    )

    print("[3/6] Creating matched NetCDF dataset ...", flush=True)
    nc = Dataset(temporary_dataset, "w", format="NETCDF4")
    nc.createDimension("time", None)
    nc.createDimension("lat", lat.size)
    nc.createDimension("lon", lon.size)
    time_var = nc.createVariable("time", "f8", ("time",))
    lat_var = nc.createVariable("lat", "f4", ("lat",))
    lon_var = nc.createVariable("lon", "f4", ("lon",))
    doy_var = nc.createVariable("doy", "i2", ("time",))
    before_var = nc.createVariable("final_before_time", "f8", ("time",))
    after_var = nc.createVariable("final_after_time", "f8", ("time",))
    weight_var = nc.createVariable("final_weight_after", "f4", ("time",))
    chunks = (1, lat.size, lon.size)
    map_variables = {}
    for name, description in (
        ("rt_vtec", "IGS combined real-time vertical TEC"),
        ("final_vtec", "IGS Final vertical TEC rotated/interpolated to RT epoch"),
        ("residual", "final_vtec - rt_vtec"),
    ):
        variable = nc.createVariable(
            name,
            "f4",
            ("time", "lat", "lon"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=chunks,
            fill_value=np.float32(np.nan),
        )
        variable.units = "TECU"
        variable.long_name = description
        map_variables[name] = variable

    time_units = "seconds since 1970-01-01 00:00:00 UTC"
    time_var.units = time_units
    time_var.calendar = "standard"
    time_var.standard_name = "time"
    lat_var.units = "degrees_north"
    lat_var.standard_name = "latitude"
    lon_var.units = "degrees_east"
    lon_var.standard_name = "longitude"
    before_var.units = time_units
    after_var.units = time_units
    weight_var.long_name = "linear interpolation weight of final_after_time"
    lat_var[:] = lat
    lon_var[:] = lon

    nc.title = "Matched IGS real-time and IGS Final global ionospheric maps"
    nc.source = "IGS combined RT-GIM (IRTG) and IGS combined Final GIM"
    nc.rt_product = "IGS combined real-time GIM (IRTG)"
    nc.final_product = "IGS combined Final GIM"
    nc.unit = "TECU"
    nc.interpolation_method = (
        "Schaer rotated-map time interpolation: each bracketing Final map is "
        "periodically longitude-shifted to the RT epoch, then linearly combined"
    )
    nc.rotation_rate_degrees_per_hour = args.rotation_rate
    nc.excluded_doys = ",".join(str(value) for value in sorted(excluded))
    nc.processing_time_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    nc.history = "Created by Code/GIM/match_rt_final.py"

    stats = {
        "global": RunningStats(),
        "low_latitude_abs_lt_30": RunningStats(),
        "mid_latitude_abs_30_to_60": RunningStats(),
        "high_latitude_abs_ge_60": RunningStats(),
    }
    latitude_masks = {
        "low_latitude_abs_lt_30": np.abs(lat) < 30.0,
        "mid_latitude_abs_30_to_60": (np.abs(lat) >= 30.0) & (np.abs(lat) < 60.0),
        "high_latitude_abs_ge_60": np.abs(lat) >= 60.0,
    }
    residual_sum = np.zeros((lat.size, lon.size), dtype=np.float64)
    residual_count = np.zeros((lat.size, lon.size), dtype=np.int64)
    latitude_sumsq = np.zeros(lat.size, dtype=np.float64)
    latitude_count = np.zeros(lat.size, dtype=np.int64)
    histogram_edges = np.linspace(-100.0, 100.0, 401)
    histogram_counts = np.zeros(histogram_edges.size - 1, dtype=np.int64)
    histogram_outside = 0
    metric_times: list[datetime] = []
    bias_by_time: list[float] = []
    rmse_by_time: list[float] = []
    samples: list[tuple[datetime, np.ndarray, np.ndarray, np.ndarray]] = []
    output_index = 0

    for candidate_index, row in enumerate(candidate_rows):
        epoch = datetime.strptime(str(row["time"]), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        rt_path = Path(str(row["rt_filepath"]))
        bracket = final_bracket(epoch, final_times, max_gap)
        assert bracket is not None
        try:
            rt_maps = read_ionex_tec_maps(rt_path)
            matches = [item for item in rt_maps if item.time == epoch]
            if len(matches) != 1:
                raise ValueError(f"expected one TEC map at inventory epoch, got {len(matches)}")
            rt_map = matches[0]
            if not np.allclose(rt_map.lat, lat, atol=1e-6):
                raise ValueError("RT and Final latitude grids differ")
            if not np.allclose(rt_map.lon, lon, atol=1e-6):
                raise ValueError("RT and Final longitude grids differ")
            final_map = interpolate_rotated_final(
                epoch,
                bracket,
                final_times,
                final_maps,
                lon,
                args.rotation_rate,
            )
            residual = final_map - rt_map.vtec
        except Exception as exc:
            row["pair_valid"] = 0
            row["reason"] = "rt_parse_or_grid_error"
            errors.append({"stage": "pair", "filepath": str(rt_path), "error": str(exc)})
            continue

        before, after, weight_after = bracket
        time_var[output_index] = date2num(epoch, units=time_units, calendar="standard")
        doy_var[output_index] = row["doy"]
        before_var[output_index] = date2num(final_times[before], units=time_units, calendar="standard")
        after_var[output_index] = date2num(final_times[after], units=time_units, calendar="standard")
        weight_var[output_index] = weight_after
        map_variables["rt_vtec"][output_index, :, :] = rt_map.vtec
        map_variables["final_vtec"][output_index, :, :] = final_map
        map_variables["residual"][output_index, :, :] = residual
        row["dataset_index"] = output_index
        stats["global"].update(rt_map.vtec, final_map)
        for name, mask in latitude_masks.items():
            stats[name].update(rt_map.vtec[mask, :], final_map[mask, :])
        finite = np.isfinite(residual)
        residual_sum[finite] += residual[finite]
        residual_count[finite] += 1
        latitude_sumsq += np.nansum(residual.astype(np.float64) ** 2, axis=1)
        latitude_count += np.sum(np.isfinite(residual), axis=1)
        values = residual[finite]
        histogram_counts += np.histogram(values, bins=histogram_edges)[0]
        histogram_outside += int(np.sum((values < histogram_edges[0]) | (values > histogram_edges[-1])))
        metric_times.append(epoch)
        bias_by_time.append(float(np.nanmean(residual)))
        rmse_by_time.append(float(np.sqrt(np.nanmean(residual.astype(np.float64) ** 2))))
        if candidate_index in sample_targets:
            samples.append(
                (epoch, rt_map.vtec.copy(), final_map.copy(), residual.copy())
            )
        output_index += 1
        if output_index % 500 == 0:
            print(f"      wrote {output_index}/{len(candidate_rows)} pairs", flush=True)

    nc.matched_pair_count = output_index
    nc.close()
    os.replace(temporary_dataset, dataset_path)
    print(f"      completed {output_index} valid pairs", flush=True)

    print("[4/6] Writing matching and quality reports ...", flush=True)
    report_fields = [
        "time",
        "year",
        "doy",
        "rt_exists",
        "rt_filepath",
        "final_before_time",
        "final_after_time",
        "final_weight_after",
        "final_interp_valid",
        "pair_valid",
        "dataset_index",
        "reason",
    ]
    write_csv(output_dir / "matching_report.csv", report, report_fields)
    for stale_split in ("train_times.csv", "val_times.csv", "test_times.csv"):
        stale_path = output_dir / stale_split
        if stale_path.exists():
            stale_path.unlink()
    warning_path = output_dir / "input_warnings.csv"
    legacy_error_path = output_dir / "input_errors.csv"
    if errors:
        write_csv(warning_path, errors, ["stage", "filepath", "error"])
    elif warning_path.exists():
        warning_path.unlink()
    if legacy_error_path.exists():
        legacy_error_path.unlink()

    metric_rows = []
    for region, accumulator in stats.items():
        metric_rows.append({"region": region, **accumulator.metrics()})
    write_csv(
        output_dir / "baseline_metrics.csv",
        metric_rows,
        ["region", "count", "bias_tecu", "mae_tecu", "rmse_tecu", "std_tecu", "correlation"],
    )

    summary = {
        "dataset": str(dataset_path),
        "year": args.year,
        "start_doy": args.start_doy,
        "end_doy": args.end_doy,
        "excluded_doys": sorted(excluded),
        "expected_20_min_epochs": len(report),
        "rt_epochs_found": len(rt_by_time),
        "matched_pairs": output_index,
        "missing_rt_epochs_including_excluded_days": sum(int(not row["rt_exists"]) for row in report),
        "excluded_existing_rt_epochs": sum(
            int(row["reason"] == "excluded_doy" and bool(row["rt_exists"])) for row in report
        ),
        "final_epoch_count": len(final_times),
        "first_final_epoch": iso_time(final_times[0]),
        "last_final_epoch": iso_time(final_times[-1]),
        "grid_shape": [int(lat.size), int(lon.size)],
        "interpolation": "rotated TEC maps followed by linear time interpolation",
        "rotation_rate_degrees_per_hour": args.rotation_rate,
        "global_baseline": stats["global"].metrics(),
        "input_warning_count": len(errors),
    }
    summary_tmp = output_dir / "processing_summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(summary_tmp, output_dir / "processing_summary.json")

    print("[5/6] Generating QC figures ...", flush=True)
    if not args.no_qc and samples:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_residual = residual_sum / residual_count
            rmse_by_latitude = np.sqrt(latitude_sumsq / latitude_count)
        generate_qc(
            output_dir,
            lat,
            lon,
            samples[len(samples) // 2],
            samples,
            mean_residual,
            histogram_edges,
            histogram_counts,
            histogram_outside,
            metric_times,
            bias_by_time,
            rmse_by_time,
            rmse_by_latitude,
        )
    print("[6/6] Done", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = create_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
