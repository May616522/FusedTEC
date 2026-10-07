"""Batch Sentinel-6 TEC extraction and comparison with IONEX GIM.

The script reads Sentinel-6A Poseidon-4 L2 LR RED NetCDF files and compares
these TEC estimates (daily observation CSV files are optional):

* below-orbit TEC converted from the raw and filtered altimeter corrections;
* the official full-column ``total_electron_content``;
* the GIM-derived correction distributed inside the Sentinel-6 product;
* an independently interpolated IONEX GIM VTEC.

Bias in every metrics table is always ``estimate - reference``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from helper.GIM_Read import parse_ionex_grid  # noqa: E402


F_KU_HZ = 13.575e9
K_IONO = 40.308
TECU = 1.0e16
TEC_FRACTION_BELOW_ORBIT = 0.881
TEC_PER_METER_KU = F_KU_HZ**2 / (K_IONO * TECU)

DEFAULT_INPUT = PROJECT_ROOT / "Data" / "Sentinel2021" / "LR_G01_reduced"
DEFAULT_IGS_GIM = PROJECT_ROOT / "Data" / "GIM2021" / "IGS" / "Final"
DEFAULT_OUTPUT = PROJECT_ROOT / "Results" / "Sentinel2021_FullYear_TEC_IGS_Final"

FILE_TIME_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")
FILE_ORBIT_RE = re.compile(r"_NT_(\d{3})_(\d{3})_")

TEC_COLUMNS = [
    "tec_below_raw_tecu",
    "tec_below_filtered_mle4_tecu",
    "tec_below_filtered_tecu",
    "tec_topside_corrected_tecu",
    "tec_official_tecu",
    "tec_product_gim_tecu",
    "tec_jpl_final_gim_tecu",
    "tec_igs_gim_tecu",
]

# (name, reference column, estimate column)
COMPARISONS = [
    (
        "igs_gim_vs_nr_below_tec",
        "tec_below_filtered_tecu",
        "tec_igs_gim_tecu",
    ),
    (
        "igs_gim_vs_total_electron_content",
        "tec_official_tecu",
        "tec_igs_gim_tecu",
    ),
]


def correction_to_tecu(correction_m: np.ndarray) -> np.ndarray:
    """Convert a Ku-band range correction (metres) to TECU."""

    return -np.asarray(correction_m, dtype=float) * TEC_PER_METER_KU


def _values(ds: xr.Dataset, name: str, size: int) -> np.ndarray:
    """Read an optional 1-Hz variable, returning NaN when it is absent."""

    if name not in ds.variables:
        return np.full(size, np.nan, dtype=float)
    return np.asarray(ds[name].values)


def _file_start(path: Path) -> datetime:
    match = FILE_TIME_RE.search(path.name)
    if not match:
        return datetime.max
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")


def _file_interval(path: Path) -> tuple[datetime, datetime]:
    """Return the filename time interval, or an unbounded fallback."""

    match = FILE_TIME_RE.search(path.name)
    if not match:
        return datetime.min, datetime.max
    return (
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S"),
        datetime.strptime(match.group(2), "%Y%m%dT%H%M%S"),
    )


def read_sentinel6_file(path: Path) -> pd.DataFrame:
    """Read all TEC and QC fields needed from one Sentinel-6 file."""

    required = {
        "time",
        "latitude",
        "longitude",
        "altitude",
        "total_electron_content",
        "iono_cor_alt",
        "iono_cor_alt_filtered",
        "iono_cor_alt_filtered_nr",
        "surface_classification_flag",
    }
    with xr.open_dataset(path, group="data_01") as ds, xr.open_dataset(
        path, group="data_01/ku"
    ) as ds_ku:
        missing = sorted(required.difference(ds.variables))
        if missing:
            raise KeyError(f"{path.name} missing variables: {missing}")
        if "iono_cor_gim" not in ds_ku.variables:
            raise KeyError(f"{path.name} missing data_01/ku/iono_cor_gim")

        time = np.asarray(ds["time"].values)
        size = time.size
        product_gim = np.asarray(ds_ku["iono_cor_gim"].values, dtype=float)
        if product_gim.size != size:
            raise ValueError(f"{path.name}: iono_cor_gim length does not match time")

        raw_correction = np.asarray(ds["iono_cor_alt"].values, dtype=float)
        filtered_mle4 = np.asarray(ds["iono_cor_alt_filtered"].values, dtype=float)
        filtered_nr = np.asarray(ds["iono_cor_alt_filtered_nr"].values, dtype=float)

        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(time),
                "lat": np.asarray(ds["latitude"].values, dtype=float),
                "lon": (
                    (np.asarray(ds["longitude"].values, dtype=float) + 180.0)
                    % 360.0
                    - 180.0
                ),
                "altitude_m": np.asarray(ds["altitude"].values, dtype=float),
                "tec_below_raw_tecu": correction_to_tecu(raw_correction),
                # The official TEC is derived from the numerical-retracker
                # filtered correction in F08/G01.  Keep MLE4 as a diagnostic,
                # but use NR for the canonical below-orbit filtered TEC.
                "tec_below_filtered_mle4_tecu": correction_to_tecu(filtered_mle4),
                "tec_below_filtered_tecu": correction_to_tecu(filtered_nr),
                "tec_official_tecu": (
                    np.asarray(ds["total_electron_content"].values, dtype=float) / TECU
                ),
                "tec_product_gim_tecu": correction_to_tecu(product_gim),
                "iono_cor_alt_m": raw_correction,
                "iono_cor_alt_filtered_mle4_m": filtered_mle4,
                "iono_cor_alt_filtered_nr_m": filtered_nr,
                "iono_cor_product_gim_m": product_gim,
                "surface_type": _values(ds, "surface_classification_flag", size),
                "rain_flag": _values(ds, "rain_flag", size),
                "rad_sea_ice_flag": _values(ds, "rad_sea_ice_flag", size),
                "manoeuvre_flag": _values(ds, "manoeuvre_flag", size),
                "pass_direction": _values(ds, "pass_direction_flag", size),
                "distance_to_coast_m": _values(ds, "distance_to_coast", size),
            }
        )

    frame["tec_topside_corrected_tecu"] = (
        frame["tec_below_filtered_tecu"] / TEC_FRACTION_BELOW_ORBIT
    )
    orbit = FILE_ORBIT_RE.search(path.name)
    frame["cycle"] = int(orbit.group(1)) if orbit else pd.NA
    frame["pass"] = int(orbit.group(2)) if orbit else pd.NA
    return frame


class IonexInterpolator:
    """Vectorized time-linear and spatial-bilinear IONEX interpolator."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def find_file(self, day: pd.Timestamp) -> Path:
        doy = day.strftime("%j")
        yy = day.strftime("%y")
        folder = self.root / doy
        candidates: list[Path] = []
        if folder.is_dir():
            candidates.extend(folder.glob(f"*{doy}*.{yy}i*"))
        if not candidates:
            candidates.extend(self.root.glob(f"**/*{doy}*.{yy}i*"))
        if not candidates:
            raise FileNotFoundError(
                f"No IONEX file for {day.date()} under {self.root}"
            )
        return sorted(candidates)[0]

    def load_day(
        self, day: pd.Timestamp
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = day.strftime("%Y%m%d")
        if key not in self._cache:
            path = self.find_file(day)
            times, lats, lons, cube = parse_ionex_grid(path)
            time_ns = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype("int64")
            lats = np.asarray(lats, dtype=float)
            lons = np.asarray(lons, dtype=float)
            cube = np.asarray(cube, dtype=float)
            cube[np.abs(cube) >= 999.0] = np.nan
            if lats[0] > lats[-1]:
                lats = lats[::-1]
                cube = cube[:, ::-1, :]
            if lons[0] > lons[-1]:
                lons = lons[::-1]
                cube = cube[:, :, ::-1]
            if len(lons) < 2 or np.any(np.diff(lons) <= 0):
                raise ValueError(f"IONEX longitude grid is not strictly increasing: {path}")

            # IONEX products normally contain either -180..180 or 0..360.
            # A product may omit the duplicated final meridian (for example,
            # 0..355).  Append it so interpolation across the date line uses
            # the first column rather than extrapolating the last grid cell.
            lon_step = float(np.median(np.diff(lons)))
            lon_span = float(lons[-1] - lons[0])
            tolerance = max(abs(lon_step) * 0.01, 1.0e-8)
            if np.isclose(lon_span + lon_step, 360.0, atol=tolerance):
                lons = np.append(lons, lons[0] + 360.0)
                cube = np.concatenate((cube, cube[:, :, :1]), axis=2)
            elif not np.isclose(lon_span, 360.0, atol=tolerance):
                raise ValueError(
                    f"IONEX longitude grid is not global (span={lon_span}): {path}"
                )
            self._cache[key] = (time_ns, lats, lons, cube)
        return self._cache[key]

    @staticmethod
    def normalize_longitudes(
        longitude: np.ndarray | pd.Series, grid_longitudes: np.ndarray
    ) -> np.ndarray:
        """Map any longitude to the convention used by an IONEX grid.

        The grid start determines the convention: a ``-180..180`` grid maps
        to ``[-180, 180)``, while a ``0..360`` grid maps to ``[0, 360)``.
        Adding or subtracting any multiple of 360 degrees gives the same
        normalized coordinate.
        """

        grid_longitudes = np.asarray(grid_longitudes, dtype=float)
        if grid_longitudes.size < 2:
            raise ValueError("IONEX longitude grid must contain at least two points")
        start = float(grid_longitudes[0])
        return (np.asarray(longitude, dtype=float) - start) % 360.0 + start

    def interpolate_day(self, frame: pd.DataFrame) -> np.ndarray:
        """Interpolate all rows in a frame belonging to one UTC day."""

        if frame.empty:
            return np.empty(0, dtype=float)
        day = pd.Timestamp(frame["time"].iloc[0]).normalize()
        time_grid, lats, lons, cube = self.load_day(day)
        target_time = frame["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        lat = frame["lat"].to_numpy(dtype=float)
        lon = self.normalize_longitudes(frame["lon"], lons)

        ti1 = np.searchsorted(time_grid, target_time, side="right")
        ti1 = np.clip(ti1, 1, len(time_grid) - 1)
        ti0 = ti1 - 1
        time_den = (time_grid[ti1] - time_grid[ti0]).astype(float)
        wt = np.divide(
            target_time - time_grid[ti0],
            time_den,
            out=np.zeros_like(time_den),
            where=time_den != 0,
        )

        yi1 = np.searchsorted(lats, lat, side="right")
        yi1 = np.clip(yi1, 1, len(lats) - 1)
        yi0 = yi1 - 1
        xi1 = np.searchsorted(lons, lon, side="right")
        xi1 = np.clip(xi1, 1, len(lons) - 1)
        xi0 = xi1 - 1
        wy = (lat - lats[yi0]) / (lats[yi1] - lats[yi0])
        wx = (lon - lons[xi0]) / (lons[xi1] - lons[xi0])

        def spatial(t_index: np.ndarray) -> np.ndarray:
            q00 = cube[t_index, yi0, xi0]
            q01 = cube[t_index, yi0, xi1]
            q10 = cube[t_index, yi1, xi0]
            q11 = cube[t_index, yi1, xi1]
            return (
                q00 * (1.0 - wy) * (1.0 - wx)
                + q01 * (1.0 - wy) * wx
                + q10 * wy * (1.0 - wx)
                + q11 * wy * wx
            )

        value0 = spatial(ti0)
        value1 = spatial(ti1)
        result = value0 + wt * (value1 - value0)
        outside = (
            (target_time < time_grid[0])
            | (target_time > time_grid[-1])
            | (lat < lats[0])
            | (lat > lats[-1])
        )
        result[outside] = np.nan
        return result

    def add_to_frame(
        self, frame: pd.DataFrame, output_column: str = "tec_external_gim_tecu"
    ) -> pd.DataFrame:
        result = frame.copy()
        result[output_column] = np.nan
        for _, index in result.groupby(result["time"].dt.normalize()).groups.items():
            result.loc[index, output_column] = self.interpolate_day(
                result.loc[index]
            )
        return result


@dataclass
class MetricAccumulator:
    n: int = 0
    sum_ref: float = 0.0
    sum_est: float = 0.0
    sum_ref2: float = 0.0
    sum_est2: float = 0.0
    sum_cross: float = 0.0
    sum_error: float = 0.0
    sum_error2: float = 0.0
    sum_abs_error: float = 0.0

    def update(self, reference: Iterable[float], estimate: Iterable[float]) -> None:
        ref = np.asarray(reference, dtype=float)
        est = np.asarray(estimate, dtype=float)
        valid = np.isfinite(ref) & np.isfinite(est)
        ref = ref[valid]
        est = est[valid]
        if ref.size == 0:
            return
        error = est - ref
        self.n += int(ref.size)
        self.sum_ref += float(ref.sum())
        self.sum_est += float(est.sum())
        self.sum_ref2 += float(np.dot(ref, ref))
        self.sum_est2 += float(np.dot(est, est))
        self.sum_cross += float(np.dot(ref, est))
        self.sum_error += float(error.sum())
        self.sum_error2 += float(np.dot(error, error))
        self.sum_abs_error += float(np.abs(error).sum())

    def result(self) -> dict[str, float | int]:
        if self.n == 0:
            return {
                "n": 0,
                **{
                    key: np.nan
                    for key in (
                        "r", "r2", "regression_r2", "slope", "intercept",
                        "bias", "rmse", "mae", "std",
                    )
                },
            }
        n = float(self.n)
        sst = self.sum_ref2 - self.sum_ref**2 / n
        var_est = self.sum_est2 - self.sum_est**2 / n
        covariance = self.sum_cross - self.sum_ref * self.sum_est / n
        r = covariance / np.sqrt(sst * var_est) if sst > 0 and var_est > 0 else np.nan
        slope = covariance / sst if sst > 0 else np.nan
        intercept = self.sum_est / n - slope * self.sum_ref / n
        error_variance = (
            self.sum_error2 - self.sum_error**2 / n
        ) / (n - 1.0) if self.n > 1 else np.nan
        return {
            "n": self.n,
            "r": r,
            "r2": 1.0 - self.sum_error2 / sst if sst > 0 else np.nan,
            "regression_r2": r**2 if np.isfinite(r) else np.nan,
            "slope": slope,
            "intercept": intercept,
            "bias": self.sum_error / n,
            "rmse": np.sqrt(self.sum_error2 / n),
            "mae": self.sum_abs_error / n,
            "std": np.sqrt(max(error_variance, 0.0)),
        }


def calculate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, reference, estimate in COMPARISONS:
        accumulator = MetricAccumulator()
        accumulator.update(frame[reference], frame[estimate])
        rows.append(
            {
                "comparison": name,
                "reference": reference,
                "estimate": estimate,
                "bias_definition": "estimate - reference",
                **accumulator.result(),
            }
        )
    return pd.DataFrame(rows)


def apply_qc(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    finite_position = np.isfinite(frame[["lat", "lon"]].to_numpy()).all(axis=1)
    if mode == "none":
        mask = finite_position
    else:
        mask = (
            finite_position
            & (frame["surface_type"] == 0)
            & np.isfinite(frame["tec_official_tecu"])
            & (frame["tec_official_tecu"] > 0.0)
        )
        if mode == "strict":
            mask &= frame["manoeuvre_flag"].isna() | (frame["manoeuvre_flag"] == 0)
            mask &= frame["rad_sea_ice_flag"].isna() | (
                frame["rad_sea_ice_flag"] == 0
            )
    return frame.loc[mask].copy()


def _sample_evenly(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    index = np.linspace(0, len(frame) - 1, maximum, dtype=int)
    return frame.iloc[index].copy()


def plot_scatter(sample: pd.DataFrame, metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    panels = [
        (
            "tec_below_filtered_tecu",
            "tec_igs_gim_tecu",
            "IGS GIM vs TEC from iono_cor_alt_filtered_nr",
            "igs_gim_vs_nr_below_tec",
        ),
        (
            "tec_official_tecu",
            "tec_igs_gim_tecu",
            "IGS GIM vs total_electron_content",
            "igs_gim_vs_total_electron_content",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    metric_lookup = metrics.set_index("comparison")
    for ax, (x_col, y_col, title, comparison) in zip(axes.ravel(), panels):
        valid = sample[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            ax.set_axis_off()
            continue
        x = valid[x_col].to_numpy()
        y = valid[y_col].to_numpy()
        ax.hexbin(x, y, gridsize=70, mincnt=1, bins="log", cmap="viridis")
        lo = float(np.nanpercentile(np.concatenate((x, y)), 0.5))
        hi = float(np.nanpercentile(np.concatenate((x, y)), 99.5))
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        if comparison in metric_lookup.index:
            metric = metric_lookup.loc[comparison]
            regression_x = np.array([lo, hi])
            regression_y = metric["slope"] * regression_x + metric["intercept"]
            ax.plot(
                regression_x,
                regression_y,
                color="red",
                linewidth=1.5,
                label="linear regression",
            )
            ax.text(
                0.03,
                0.97,
                (
                    f"y={metric['slope']:.3f}x{metric['intercept']:+.3f}\n"
                    f"R={metric['r']:.3f}, regression R2={metric['regression_r2']:.3f}\n"
                    f"agreement R2={metric['r2']:.3f}\n"
                    f"RMSE={metric['rmse']:.3f}, Bias={metric['bias']:.3f}"
                ),
                transform=ax.transAxes,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        ax.set(xlabel=x_col, ylabel=y_col, title=title)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right")
    fig.suptitle("IGS GIM regression against Sentinel-6 TEC (ocean QC)")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_daily_means(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    columns = [
        ("tec_igs_gim_tecu_mean", "IGS GIM"),
        ("tec_jpl_final_gim_tecu_mean", "JPL FINAL GIM"),
        ("tec_official_tecu_mean", "total_electron_content"),
        ("tec_below_filtered_tecu_mean", "iono_cor_alt_filtered_nr TEC"),
    ]
    dates = pd.to_datetime(summary["date"])
    for column, label in columns:
        if column in summary and summary[column].notna().any():
            ax.plot(dates, summary[column], marker="o", markersize=3, label=label)
    ax.set(
        xlabel="UTC date",
        ylabel="Daily mean TEC (TECU)",
        title="Daily mean TEC: IGS, JPL FINAL and Sentinel-6",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_daily_metrics(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    selected = metrics[metrics["comparison"].isin(name for name, _, _ in COMPARISONS)].copy()
    if selected.empty:
        return
    labels = {
        "igs_gim_vs_nr_below_tec": "IGS GIM - NR below TEC",
        "igs_gim_vs_total_electron_content": "IGS GIM - total_electron_content",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for name, group in selected.groupby("comparison"):
        group = group.sort_values("date")
        dates = pd.to_datetime(group["date"])
        label = labels.get(name, name)
        axes[0].plot(dates, group["r2"], marker="o", markersize=3, label=label)
        axes[1].plot(dates, group["rmse"], marker="o", markersize=3, label=label)
        axes[2].plot(dates, group["bias"], marker="o", markersize=3, label=label)
    axes[0].set_ylabel("R2")
    axes[1].set_ylabel("RMSE (TECU)")
    axes[2].set_ylabel("Bias (TECU)\nestimate - reference")
    axes[2].set_xlabel("UTC date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Daily IGS GIM and Sentinel-6 TEC comparison metrics")
    axes[2].tick_params(axis="x", labelrotation=30)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_daily_bias_rmse(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """Plot the requested daily bias and RMSE series for both comparisons."""

    selected = metrics[
        metrics["comparison"].isin(name for name, _, _ in COMPARISONS)
    ].copy()
    if selected.empty:
        return
    labels = {
        "igs_gim_vs_nr_below_tec": "IGS GIM - NR below-orbit TEC",
        "igs_gim_vs_total_electron_content": (
            "IGS GIM - total_electron_content"
        ),
    }
    colors = {
        "igs_gim_vs_nr_below_tec": "tab:blue",
        "igs_gim_vs_total_electron_content": "tab:orange",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for name, group in selected.groupby("comparison"):
        group = group.sort_values("date")
        dates = pd.to_datetime(group["date"])
        label = labels.get(name, name)
        color = colors.get(name)
        axes[0].plot(dates, group["bias"], linewidth=1.2, label=label, color=color)
        axes[1].plot(dates, group["rmse"], linewidth=1.2, label=label, color=color)
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Daily bias (TECU)\nIGS GIM - Sentinel-6")
    axes[1].set_ylabel("Daily RMSE (TECU)")
    axes[1].set_xlabel("UTC date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Daily IGS Final GIM and Sentinel-6 TEC differences")
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_daily_bias_rmse_distribution(
    metrics: pd.DataFrame, output: Path, dpi: int
) -> None:
    """Plot across-day distributions of daily bias and daily RMSE."""

    selected = metrics[
        metrics["comparison"].isin(name for name, _, _ in COMPARISONS)
    ].copy()
    if selected.empty:
        return
    labels = {
        "igs_gim_vs_nr_below_tec": "IGS - NR below-orbit TEC",
        "igs_gim_vs_total_electron_content": "IGS - total_electron_content",
    }
    colors = {
        "igs_gim_vs_nr_below_tec": "tab:blue",
        "igs_gim_vs_total_electron_content": "tab:orange",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, group in selected.groupby("comparison"):
        label = labels.get(name, name)
        color = colors.get(name)
        for ax, metric, title in (
            (axes[0], "bias", "Distribution of daily bias"),
            (axes[1], "rmse", "Distribution of daily RMSE"),
        ):
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            bins = min(35, max(8, int(np.sqrt(values.size))))
            ax.hist(values, bins=bins, alpha=0.45, density=True, label=label, color=color)
            ax.axvline(
                np.mean(values), color=color, linewidth=1.5, linestyle="--"
            )
            ax.set_title(title)
            ax.set_xlabel(f"Daily {metric} (TECU)")
            ax.set_ylabel("Density")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Across-day distributions (dashed lines show means)")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def process(args: argparse.Namespace) -> None:
    start = pd.Timestamp(args.start_date)
    end_exclusive = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    files = sorted(args.input_dir.glob(args.pattern), key=_file_start)
    # NetCDF names contain precise coverage bounds.  Filter before opening the
    # files so a one-day or one-month run does not scan the entire annual set.
    files = [
        path
        for path in files
        if pd.Timestamp(_file_interval(path)[1]) > start
        and pd.Timestamp(_file_interval(path)[0]) < end_exclusive
    ]
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.pattern} and requested dates in {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = args.output_dir / "daily"
    figure_dir = args.output_dir / "figures"
    daily_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    if args.no_external_gim:
        jpl_interpolator = None
        igs_interpolator = None
    else:
        # JPL is optional.  The requested annual analysis depends only on the
        # explicitly supplied IGS Final archive.
        jpl_interpolator = (
            IonexInterpolator(args.jpl_gim_dir)
            if args.jpl_gim_dir is not None
            else None
        )
        igs_interpolator = IonexInterpolator(args.igs_gim_dir)
    buffers: dict[pd.Timestamp, list[pd.DataFrame]] = {}
    audits: list[dict[str, object]] = []
    daily_summaries: list[dict[str, object]] = []
    daily_metric_frames: list[pd.DataFrame] = []
    plot_samples: list[pd.DataFrame] = []
    global_metrics = {name: MetricAccumulator() for name, _, _ in COMPARISONS}

    def flush_day(day: pd.Timestamp) -> None:
        parts = buffers.pop(day, [])
        if not parts:
            return
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.sort_values(["time", "lat", "lon"]).drop_duplicates(
            ["time", "lat", "lon"], keep="first"
        )
        frame["residual_below_filtered_minus_official_tecu"] = (
            frame["tec_below_filtered_tecu"] - frame["tec_official_tecu"]
        )
        frame["residual_product_gim_minus_official_tecu"] = (
            frame["tec_product_gim_tecu"] - frame["tec_official_tecu"]
        )
        frame["residual_jpl_final_gim_minus_official_tecu"] = (
            frame["tec_jpl_final_gim_tecu"] - frame["tec_official_tecu"]
        )
        frame["residual_igs_gim_minus_official_tecu"] = (
            frame["tec_igs_gim_tecu"] - frame["tec_official_tecu"]
        )
        # Keep the research-plan convention (Sentinel-6 minus GIM) as an
        # explicit companion to the metric convention (estimate minus
        # reference).  The names make the sign unambiguous.
        frame["residual_official_minus_product_gim_tecu"] = -frame[
            "residual_product_gim_minus_official_tecu"
        ]
        frame["residual_official_minus_jpl_final_gim_tecu"] = -frame[
            "residual_jpl_final_gim_minus_official_tecu"
        ]
        frame["residual_official_minus_igs_gim_tecu"] = -frame[
            "residual_igs_gim_minus_official_tecu"
        ]
        frame["residual_igs_gim_minus_below_filtered_tecu"] = (
            frame["tec_igs_gim_tecu"] - frame["tec_below_filtered_tecu"]
        )

        date_text = day.strftime("%Y%m%d")
        if args.write_daily_observations:
            frame.to_csv(
                daily_dir / f"sentinel6_tec_{date_text}.csv",
                index=False,
                float_format="%.6f",
                date_format="%Y-%m-%dT%H:%M:%S",
            )
        means = frame[TEC_COLUMNS].mean(numeric_only=True)
        daily_summaries.append(
            {
                "date": day.date().isoformat(),
                "n": len(frame),
                **{f"{column}_mean": means.get(column, np.nan) for column in TEC_COLUMNS},
            }
        )
        metric_frame = calculate_metrics(frame)
        metric_frame.insert(0, "date", day.date().isoformat())
        daily_metric_frames.append(metric_frame)
        for name, reference, estimate in COMPARISONS:
            global_metrics[name].update(frame[reference], frame[estimate])
        plot_samples.append(_sample_evenly(frame, args.plot_sample_per_day))

    for index, path in enumerate(files, start=1):
        try:
            raw = read_sentinel6_file(path)
            in_window = raw[(raw["time"] >= start) & (raw["time"] < end_exclusive)].copy()
            selected = apply_qc(in_window, args.qc)
            selected["tec_jpl_final_gim_tecu"] = np.nan
            selected["tec_igs_gim_tecu"] = np.nan
            if jpl_interpolator is not None:
                selected = jpl_interpolator.add_to_frame(
                    selected, "tec_jpl_final_gim_tecu"
                )
            if igs_interpolator is not None:
                selected = igs_interpolator.add_to_frame(
                    selected, "tec_igs_gim_tecu"
                )
            for day, group in selected.groupby(selected["time"].dt.normalize()):
                buffers.setdefault(pd.Timestamp(day), []).append(group)
            audits.append(
                {
                    "file": path.name,
                    "status": "ok",
                    "input_records": len(raw),
                    "records_in_date_window": len(in_window),
                    "records_after_qc": len(selected),
                    "jpl_gim_valid": int(selected["tec_jpl_final_gim_tecu"].notna().sum()),
                    "igs_gim_valid": int(selected["tec_igs_gim_tecu"].notna().sum()),
                    "error": "",
                }
            )
        except Exception as exc:  # continue the month and retain an audit trail
            audits.append(
                {
                    "file": path.name,
                    "status": "error",
                    "input_records": 0,
                    "records_in_date_window": 0,
                    "records_after_qc": 0,
                    "jpl_gim_valid": 0,
                    "igs_gim_valid": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        next_day = None
        if index < len(files):
            next_start = _file_start(files[index])
            if next_start != datetime.max:
                next_day = pd.Timestamp(next_start).normalize()
        flushable = [day for day in buffers if next_day is None or day < next_day]
        for day in sorted(flushable):
            flush_day(day)
        if index % 25 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} files", flush=True)

    for day in sorted(list(buffers)):
        flush_day(day)

    audit = pd.DataFrame(audits)
    audit.to_csv(args.output_dir / "file_audit.csv", index=False, encoding="utf-8-sig")
    daily_summary = pd.DataFrame(daily_summaries).sort_values("date")
    daily_summary.to_csv(args.output_dir / "daily_summary.csv", index=False)
    daily_metrics = pd.concat(daily_metric_frames, ignore_index=True)
    daily_metrics.to_csv(args.output_dir / "daily_metrics.csv", index=False)

    monthly_rows = []
    for name, reference, estimate in COMPARISONS:
        monthly_rows.append(
            {
                "comparison": name,
                "reference": reference,
                "estimate": estimate,
                "bias_definition": "estimate - reference",
                **global_metrics[name].result(),
            }
        )
    overall_metrics = pd.DataFrame(monthly_rows)
    overall_metrics.to_csv(args.output_dir / "annual_metrics.csv", index=False)

    sample = pd.concat(plot_samples, ignore_index=True) if plot_samples else pd.DataFrame()
    if not sample.empty:
        plot_scatter(
            sample, overall_metrics, figure_dir / "igs_tec_regression.png", args.dpi
        )
        plot_daily_means(
            daily_summary, figure_dir / "daily_mean_four_sources.png", args.dpi
        )
    plot_daily_bias_rmse(
        daily_metrics, figure_dir / "daily_bias_rmse.png", args.dpi
    )
    plot_daily_bias_rmse_distribution(
        daily_metrics,
        figure_dir / "daily_bias_rmse_distribution.png",
        args.dpi,
    )
    plot_daily_metrics(
        daily_metrics, figure_dir / "daily_r2_rmse_bias.png", args.dpi
    )

    print(f"Output: {args.output_dir.resolve()}")
    print(overall_metrics.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Sentinel-6 TEC extraction and external IONEX comparison"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--jpl-gim-dir", "--gim-dir", dest="jpl_gim_dir", type=Path,
        default=None,
    )
    parser.add_argument("--igs-gim-dir", type=Path, default=DEFAULT_IGS_GIM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pattern", default="*.nc")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--qc", choices=("basic", "strict", "none"), default="basic")
    parser.add_argument("--no-external-gim", action="store_true")
    parser.add_argument(
        "--write-daily-observations",
        action="store_true",
        help="write large observation-level daily CSV files (off by default)",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--plot-sample-per-day", type=int, default=1500)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


if __name__ == "__main__":
    process(build_parser().parse_args())
