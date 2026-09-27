"""按全球、陆地和海洋区域分析 COSMIC TEC 与 GIM VTEC 的一致性。

脚本比较三组关系：tec0 与 GIM、tec0 与 tec1，以及 (tec0+tec1) 与 GIM。
``tec1 == -999`` 的记录视为无效；默认对每个比较关系和区域分别剔除绝对
残差最大的 0.1%，随后输出总体、逐日和空间统计表及诊断图。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    ccrs = None
    HAS_CARTOPY = False


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim"
DEFAULT_LANDMASK = PROJECT_ROOT / "Data" / "Other" / "landmask_static.nc"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Cosmic2021_TEC_GIM_land_ocean_analysis"
FILE_PATTERN = re.compile(r"^cosmic2_ionPrF_(\d{4})_(\d{3})\.csv$", re.IGNORECASE)
REQUIRED_COLUMNS = ("time", "lat", "lon", "tec0", "tec1", "gim_vtec")
REGION_ORDER = ("all", "land", "ocean")
REGION_LABELS = {"all": "Overall", "land": "Land", "ocean": "Ocean"}
REGION_COLORS = {"all": "#222222", "land": "#b24a33", "ocean": "#2878b5"}


@dataclass(frozen=True)
class Comparison:
    """描述一组 TEC 比较所需的字段、标签和偏差定义。"""

    key: str
    title: str
    x_column: str
    y_column: str
    x_label: str
    y_label: str
    bias_definition: str


COMPARISONS = (
    Comparison(
        "tec0_vs_gim",
        "COSMIC tec0 vs GIM VTEC",
        "tec0",
        "gim_vtec",
        "tec0 (TECU)",
        "GIM VTEC (TECU)",
        "gim_vtec - tec0",
    ),
    Comparison(
        "tec0_vs_tec1",
        "COSMIC tec0 vs tec1",
        "tec0",
        "tec1",
        "tec0 (TECU)",
        "tec1 (TECU)",
        "tec1 - tec0",
    ),
    Comparison(
        "cosmic_total_vs_gim",
        "COSMIC (tec0 + tec1) vs GIM VTEC",
        "cosmic_total_tec",
        "gim_vtec",
        "tec0 + tec1 (TECU)",
        "GIM VTEC (TECU)",
        "gim_vtec - (tec0 + tec1)",
    ),
)


@dataclass(frozen=True)
class LandMask:
    """保存规则经纬度陆地比例网格及陆海判别阈值。"""

    latitude: np.ndarray
    longitude: np.ndarray
    land_fraction: np.ndarray
    threshold: float
    source: Path
    variable: str

    def fraction_at(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """用最近邻网格查询指定经纬度处的陆地比例。"""

        latitude = np.asarray(lat, dtype=float)
        longitude = np.mod(np.asarray(lon, dtype=float), 360.0)
        lat_index = _nearest_indices(self.latitude, latitude)
        lon_index = _nearest_indices(self.longitude, longitude)
        return self.land_fraction[lat_index, lon_index]

    def is_land(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.fraction_at(lat, lon) >= self.threshold


def _nearest_indices(coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
    """返回每个目标值在有序坐标轴上的最近网格索引。"""

    right = np.searchsorted(coordinates, values, side="left")
    right = np.clip(right, 1, len(coordinates) - 1)
    left = right - 1
    choose_right = np.abs(coordinates[right] - values) < np.abs(
        values - coordinates[left]
    )
    return np.where(choose_right, right, left)


def load_landmask(path: Path, threshold: float) -> LandMask:
    """读取陆地比例 NetCDF，并统一经度顺序和纬度方向。"""

    with xr.open_dataset(path) as dataset:
        if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
            raise ValueError("陆海掩膜缺少 latitude/longitude 坐标")
        if "lsm" in dataset.data_vars:
            variable = "lsm"
        else:
            candidates = [
                name
                for name, value in dataset.data_vars.items()
                if {"latitude", "longitude"}.issubset(value.dims)
            ]
            if not candidates:
                raise ValueError("陆海掩膜中找不到二维 latitude/longitude 数据变量")
            variable = candidates[0]
        field = dataset[variable].transpose("latitude", "longitude")
        latitude = np.asarray(dataset["latitude"].values, dtype=float)
        longitude = np.mod(np.asarray(dataset["longitude"].values, dtype=float), 360.0)
        values = np.asarray(field.values, dtype=float)

    lat_order = np.argsort(latitude)
    lon_order = np.argsort(longitude)
    latitude = latitude[lat_order]
    longitude = longitude[lon_order]
    values = values[np.ix_(lat_order, lon_order)]
    if not np.isfinite(values).all():
        raise ValueError("陆海掩膜包含缺失值，无法可靠分类")
    return LandMask(latitude, longitude, values, threshold, path.resolve(), variable)


# 按文件名模式和可选年积日集合发现待分析的日 CSV。
def discover_files(
    input_dir: Path, year: int, start_doy: int, end_doy: int
) -> list[tuple[int, Path]]:
    selected: list[tuple[int, Path]] = []
    for path in sorted(input_dir.glob("cosmic2_ionPrF_*.csv")):
        if path.name.lower().endswith("_qc_log.csv"):
            continue
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        file_year, doy = int(match.group(1)), int(match.group(2))
        if file_year == year and start_doy <= doy <= end_doy:
            selected.append((doy, path))
    return selected


# 合并输入数据，清洗时间与数值字段，并为每条记录标记陆地或海洋。
def load_and_filter(
    files: list[tuple[int, Path]], year: int, landmask: LandMask
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    raw_rows = 0
    invalid_tec1_rows = 0
    invalid_numeric_rows = 0
    invalid_position_rows = 0
    invalid_time_rows = 0

    for doy, path in files:
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        frame["source_doy"] = doy
        frame["source_file"] = path.name
        raw_rows += len(frame)

        numeric_columns = ["lat", "lon", "tec0", "tec1", "gim_vtec"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid_tec1 = np.isclose(
            frame["tec1"].to_numpy(dtype=float), -999.0, rtol=0.0, atol=1.0e-9
        )
        invalid_tec1_rows += int(invalid_tec1.sum())
        frame = frame.loc[~invalid_tec1].copy()

        numeric_valid = np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all(axis=1)
        invalid_numeric_rows += int((~numeric_valid).sum())
        frame = frame.loc[numeric_valid].copy()
        frame["lon"] = (frame["lon"] + 180.0) % 360.0 - 180.0
        position_valid = frame["lat"].between(-90.0, 90.0) & frame["lon"].between(
            -180.0, 180.0, inclusive="left"
        )
        invalid_position_rows += int((~position_valid).sum())
        frame = frame.loc[position_valid].copy()
        frame["datetime"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        time_valid = frame["datetime"].notna()
        invalid_time_rows += int((~time_valid).sum())
        frames.append(frame.loc[time_valid].copy())

    if not frames:
        raise ValueError("筛选后没有可分析的数据")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(
        data["source_doy"] - 1, unit="D"
    )
    data["cosmic_total_tec"] = data["tec0"] + data["tec1"]
    land = landmask.is_land(data["lat"].to_numpy(), data["lon"].to_numpy())
    data["region"] = np.where(land, "land", "ocean")
    data = data.sort_values("datetime", kind="stable").reset_index(drop=True)
    filter_summary = pd.DataFrame(
        [
            {"item": "input_files", "count": len(files)},
            {"item": "raw_rows", "count": raw_rows},
            {"item": "removed_tec1_minus_999", "count": invalid_tec1_rows},
            {"item": "removed_nonfinite_numeric", "count": invalid_numeric_rows},
            {"item": "removed_invalid_position", "count": invalid_position_rows},
            {"item": "removed_invalid_time", "count": invalid_time_rows},
            {"item": "analysis_rows", "count": len(data)},
            {"item": "land_rows", "count": int((data["region"] == "land").sum())},
            {"item": "ocean_rows", "count": int((data["region"] == "ocean").sum())},
        ]
    )
    return data, filter_summary


def calculate_metrics(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    """计算样本数、偏差、MAE、RMSE、相关系数和线性回归参数。"""

    x_values = x.to_numpy(dtype=np.float64)
    y_values = y.to_numpy(dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[valid], y_values[valid]
    count = int(x_values.size)
    if count == 0:
        return dict(count=0, r=np.nan, bias=np.nan, rmse=np.nan, mean_x=np.nan, mean_y=np.nan)
    residual = y_values - x_values
    correlation = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if count >= 2 and np.std(x_values) > 0.0 and np.std(y_values) > 0.0
        else np.nan
    )
    return {
        "count": count,
        "r": correlation,
        "bias": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_x": float(np.mean(x_values)),
        "mean_y": float(np.mean(y_values)),
    }


def region_frames(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """把数据拆分为全球、陆地和海洋三个视图。"""

    return {
        "all": data,
        "land": data.loc[data["region"] == "land"],
        "ocean": data.loc[data["region"] == "ocean"],
    }


# 按绝对残差分位数剔除异常值，并同时保留被剔除记录用于审计。
def filter_comparison_outliers(
    data: pd.DataFrame, outlier_fraction: float
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    summary_rows: list[dict[str, object]] = []
    for region, regional_data in region_frames(data).items():
        outputs[region] = {}
        for comparison in COMPARISONS:
            absolute_residual = (
                regional_data[comparison.y_column] - regional_data[comparison.x_column]
            ).abs()
            if outlier_fraction == 0.0 or regional_data.empty:
                cutoff = float("inf")
                retained = regional_data.copy()
            else:
                cutoff = float(absolute_residual.quantile(1.0 - outlier_fraction))
                retained = regional_data.loc[absolute_residual <= cutoff].copy()
            outputs[region][comparison.key] = retained
            summary_rows.append(
                {
                    "region": region,
                    "comparison": comparison.key,
                    "method": "largest absolute residual fraction",
                    "outlier_fraction": outlier_fraction,
                    "absolute_residual_cutoff_tecu": cutoff,
                    "raw_count": len(regional_data),
                    "removed_count": len(regional_data) - len(retained),
                    "retained_count": len(retained),
                }
            )
    return outputs, pd.DataFrame(summary_rows)


# 对每种比较关系和区域计算原始/筛选后的总体指标。
def overall_statistics(
    data: pd.DataFrame,
    comparison_data: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    raw_regions = region_frames(data)
    for region in REGION_ORDER:
        for comparison in COMPARISONS:
            selected = raw_regions[region] if comparison_data is None else comparison_data[region][comparison.key]
            row: dict[str, object] = {
                "region": region,
                "comparison": comparison.key,
                "x_variable": comparison.x_column,
                "y_variable": comparison.y_column,
                "bias_definition": comparison.bias_definition,
            }
            row.update(calculate_metrics(selected[comparison.x_column], selected[comparison.y_column]))
            rows.append(row)
    return pd.DataFrame(rows)


def daily_statistics(comparison_data: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """按日期、比较关系和区域汇总统计指标。"""

    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        for comparison in COMPARISONS:
            for date_value, group in comparison_data[region][comparison.key].groupby("date", sort=True):
                row: dict[str, object] = {
                    "date": date_value,
                    "region": region,
                    "comparison": comparison.key,
                    "bias_definition": comparison.bias_definition,
                }
                row.update(calculate_metrics(group[comparison.x_column], group[comparison.y_column]))
                rows.append(row)
    return pd.DataFrame(rows)


def add_spatial_bins(data: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """按给定角度分辨率为观测添加经纬度网格中心。"""

    result = data.copy()
    lon_index = np.floor((result["lon"] + 180.0) / grid_size).astype(int)
    lat_for_bin = np.minimum(result["lat"].to_numpy(), np.nextafter(90.0, -np.inf))
    lat_index = np.floor((lat_for_bin + 90.0) / grid_size).astype(int)
    max_lon_index = int(round(360.0 / grid_size)) - 1
    max_lat_index = int(round(180.0 / grid_size)) - 1
    result["lon_index"] = np.clip(lon_index, 0, max_lon_index)
    result["lat_index"] = np.clip(lat_index, 0, max_lat_index)
    result["lon_center"] = -180.0 + (result["lon_index"] + 0.5) * grid_size
    result["lat_center"] = -90.0 + (result["lat_index"] + 0.5) * grid_size
    return result


# 在每个经纬度网格中计算样本量、偏差、误差和相关性。
def spatial_statistics(
    comparison_data: dict[str, dict[str, pd.DataFrame]], grid_size: float, min_bin_count: int
) -> dict[tuple[str, str], pd.DataFrame]:
    outputs: dict[tuple[str, str], pd.DataFrame] = {}
    for region in REGION_ORDER:
        for comparison in COMPARISONS:
            binned = add_spatial_bins(comparison_data[region][comparison.key], grid_size)
            grouped = binned.groupby(["lat_index", "lon_index", "lat_center", "lon_center"], sort=True)
            rows: list[dict[str, object]] = []
            for (lat_index, lon_index, lat_center, lon_center), group in grouped:
                metrics = calculate_metrics(group[comparison.x_column], group[comparison.y_column])
                row: dict[str, object] = {
                    "region": region,
                    "comparison": comparison.key,
                    "lat_index": int(lat_index),
                    "lon_index": int(lon_index),
                    "lat_center": float(lat_center),
                    "lon_center": float(lon_center),
                    "meets_min_count": int(metrics["count"]) >= min_bin_count,
                }
                row.update(metrics)
                if int(metrics["count"]) < min_bin_count:
                    row.update(r=np.nan, bias=np.nan, rmse=np.nan)
                rows.append(row)
            outputs[(region, comparison.key)] = pd.DataFrame(rows)
    return outputs


def _metric_text(row: pd.Series) -> str:
    return (
        f"N = {int(row['count']):,}\nR = {row['r']:.4f}\n"
        f"Bias = {row['bias']:.4f} TECU\nRMSE = {row['rmse']:.4f} TECU"
    )


# 绘制全球、陆地和海洋三类散点关系及其回归统计。
def plot_scatter_by_region(
    comparison_data: dict[str, dict[str, pd.DataFrame]],
    overall: pd.DataFrame,
    comparison: Comparison,
    outlier_fraction: float,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
    for axis, region in zip(axes, REGION_ORDER):
        data = comparison_data[region][comparison.key]
        x = data[comparison.x_column].to_numpy(dtype=float)
        y = data[comparison.y_column].to_numpy(dtype=float)
        limits_source = np.concatenate((x, y))
        low, high = np.nanpercentile(limits_source, [0.5, 99.5])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(limits_source)), float(np.nanmax(limits_source))
        padding = max((high - low) * 0.04, 0.1)
        low, high = float(low - padding), float(high + padding)
        density = axis.hexbin(
            x, y, gridsize=75, bins="log", mincnt=1,
            extent=(low, high, low, high), cmap="viridis",
        )
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1.0)
        axis.set(xlim=(low, high), ylim=(low, high), xlabel=comparison.x_label, ylabel=comparison.y_label)
        axis.set_title(REGION_LABELS[region])
        axis.grid(True, alpha=0.22)
        metric_row = overall.loc[
            (overall["comparison"] == comparison.key) & (overall["region"] == region)
        ].iloc[0]
        axis.text(
            0.03, 0.97, _metric_text(metric_row), transform=axis.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.5"},
        )
        fig.colorbar(density, ax=axis, pad=0.02).set_label("log10 observation count")
    fig.suptitle(
        f"{comparison.title}\nLargest {outlier_fraction:.2%} absolute residuals removed per region"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_daily_metrics(daily: pd.DataFrame, output_path: Path, dpi: int) -> None:
    """绘制各区域偏差、RMSE 和相关系数的逐日变化。"""

    fig, axes = plt.subplots(
        len(COMPARISONS), 3, figsize=(16, 4.2 * len(COMPARISONS)),
        sharex="col", constrained_layout=True,
    )
    metric_specs = (("r", "Pearson R"), ("bias", "Bias (TECU)"), ("rmse", "RMSE (TECU)"))
    for row_index, comparison in enumerate(COMPARISONS):
        for column_index, (metric, label) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            for region in REGION_ORDER:
                subset = daily.loc[
                    (daily["comparison"] == comparison.key) & (daily["region"] == region)
                ].sort_values("date")
                axis.plot(
                    subset["date"], subset[metric], marker="o", markersize=2.5,
                    linewidth=1.15, color=REGION_COLORS[region], label=REGION_LABELS[region],
                )
            if metric == "r":
                axis.set_ylim(-1.05, 1.05)
            elif metric == "bias":
                axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.28)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            axis.tick_params(axis="x", rotation=35)
            if column_index == 1:
                axis.set_title(comparison.title, pad=10)
            if row_index == 0 and column_index == 2:
                axis.legend(loc="best", frameon=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _grid_values(
    spatial: pd.DataFrame, metric: str, grid_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_centers = np.arange(-180.0 + grid_size / 2.0, 180.0, grid_size)
    lat_centers = np.arange(-90.0 + grid_size / 2.0, 90.0, grid_size)
    grid = np.full((len(lat_centers), len(lon_centers)), np.nan)
    if not spatial.empty:
        grid[
            spatial["lat_index"].to_numpy(dtype=int),
            spatial["lon_index"].to_numpy(dtype=int),
        ] = spatial[metric].to_numpy(dtype=float)
    return lon_centers, lat_centers, grid


def _smooth_weighted_grid(
    values: np.ndarray, counts: np.ndarray, sigma: float, support_threshold: float = 0.04
) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0.0)
    if not valid.any() or sigma == 0.0:
        return values.copy()
    weights = np.where(valid, counts, 0.0)
    numerator = gaussian_filter(
        np.where(valid, values * weights, 0.0), sigma, mode=("nearest", "wrap")
    )
    denominator = gaussian_filter(weights, sigma, mode=("nearest", "wrap"))
    support = gaussian_filter(valid.astype(float), sigma, mode=("nearest", "wrap"))
    result = np.full_like(values, np.nan, dtype=float)
    use = (denominator > 0.0) & (support >= support_threshold)
    result[use] = numerator[use] / denominator[use]
    return result


def _robust_positive_max(values: np.ndarray, percentile: float = 98.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    result = float(np.percentile(finite, percentile))
    return result if result > 0.0 else 1.0


def _contour_levels(metric: str, grid: np.ndarray) -> np.ndarray:
    if metric == "r":
        return np.linspace(-1.0, 1.0, 21)
    if metric == "bias":
        limit = _robust_positive_max(np.abs(grid))
        return np.linspace(-limit, limit, 21)
    percentile = 99.0 if metric == "count" else 98.0
    return np.linspace(0.0, _robust_positive_max(grid, percentile), 21)


# 绘制偏差、RMSE、相关系数和样本数的全球空间分布。
def plot_spatial_metrics(
    spatial: pd.DataFrame,
    comparison: Comparison,
    region: str,
    landmask: LandMask,
    grid_size: float,
    min_bin_count: int,
    smooth_sigma: float,
    output_path: Path,
    dpi: int,
) -> None:
    subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 7.8), subplot_kw=subplot_kw, constrained_layout=True)
    metric_specs = (
        ("count", "Observation count", "viridis"),
        ("r", f"Pearson R (raw cell N >= {min_bin_count})", "coolwarm"),
        ("bias", f"Bias (TECU, raw cell N >= {min_bin_count})", "RdBu_r"),
        ("rmse", f"RMSE (TECU, raw cell N >= {min_bin_count})", "magma"),
    )
    lon_centers, lat_centers, counts = _grid_values(spatial, "count", grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    region_mask = np.ones_like(lon_mesh, dtype=bool)
    if region != "all":
        grid_is_land = landmask.is_land(lat_mesh.ravel(), lon_mesh.ravel()).reshape(lat_mesh.shape)
        region_mask = grid_is_land if region == "land" else ~grid_is_land
    if spatial.empty:
        lat_min, lat_max = -60.0, 60.0
    else:
        lat_min = max(-90.0, float(spatial["lat_center"].min()) - 1.5 * grid_size)
        lat_max = min(90.0, float(spatial["lat_center"].max()) + 1.5 * grid_size)

    for axis, (metric, title, cmap) in zip(axes.flat, metric_specs):
        _, _, raw_grid = _grid_values(spatial, metric, grid_size)
        smoothing_weights = (
            np.where(np.isfinite(raw_grid), 1.0, 0.0) if metric == "count" else counts
        )
        grid = _smooth_weighted_grid(raw_grid, smoothing_weights, smooth_sigma)
        grid[~region_mask] = np.nan
        if metric == "r":
            grid = np.clip(grid, -1.0, 1.0)
        plot_kwargs: dict[str, object] = {
            "levels": _contour_levels(metric, grid),
            "cmap": cmap,
            "extend": "both" if metric in {"bias", "r"} else "max",
        }
        if HAS_CARTOPY:
            plot_kwargs["transform"] = ccrs.PlateCarree()
        image = axis.contourf(lon_centers, lat_centers, grid, **plot_kwargs)
        if HAS_CARTOPY:
            axis.set_extent([-180.0, 180.0, lat_min, lat_max], crs=ccrs.PlateCarree())
            axis.coastlines(resolution="110m", linewidth=0.65, color="0.2")
            gridlines = axis.gridlines(
                draw_labels=True, linewidth=0.35, color="0.4", alpha=0.45, linestyle="--"
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
        else:
            axis.set(xlim=(-180, 180), ylim=(lat_min, lat_max), xlabel="Longitude", ylabel="Latitude")
            axis.grid(True, alpha=0.25)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, orientation="vertical", pad=0.025).set_label(title)
    fig.suptitle(
        f"{REGION_LABELS[region]} spatial metrics: {comparison.title}\n"
        f"Count-weighted Gaussian smoothing: sigma={smooth_sigma:g} grid cells; "
        f"bias = {comparison.bias_definition}"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# 写出数据筛选、偏差定义和统计方法说明，保证结果可追溯。
def write_method_note(
    output_path: Path,
    landmask: LandMask,
    grid_size: float,
    min_bin_count: int,
    smooth_sigma: float,
    outlier_fraction: float,
) -> None:
    output_path.write_text(
        "\n".join(
            [
                "COSMIC TEC / GIM analysis method",
                "",
                "Regions: all, land, ocean.",
                f"Land mask: {landmask.source}",
                f"Land-mask variable: {landmask.variable}",
                f"Land definition: land fraction >= {landmask.threshold:g}",
                "Invalid rule: rows with tec1 == -999 are removed before analysis.",
                "Robust rule: for each comparison and region separately, remove the "
                f"largest {outlier_fraction:.4%} absolute residuals.",
                f"Raw spatial grid size: {grid_size:g} degrees.",
                f"Minimum raw grid count for R/Bias/RMSE: {min_bin_count}.",
                "Spatial figure only: count-weighted Gaussian smoothing with "
                f"sigma={smooth_sigma:g} grid cells. CSV grids remain unsmoothed.",
                "Bias is always y - x; see bias_definition in each CSV.",
            ]
        ),
        encoding="utf-8",
    )


# 统一保存统计表、异常值记录、方法说明以及全部图件。
def write_outputs(
    data: pd.DataFrame,
    comparison_data: dict[str, dict[str, pd.DataFrame]],
    filter_summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    raw_overall: pd.DataFrame,
    overall: pd.DataFrame,
    daily: pd.DataFrame,
    spatial_outputs: dict[tuple[str, str], pd.DataFrame],
    landmask: LandMask,
    output_dir: Path,
    grid_size: float,
    min_bin_count: int,
    smooth_sigma: float,
    outlier_fraction: float,
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    spatial_dir = output_dir / "spatial_csv"
    figures_dir.mkdir(exist_ok=True)
    spatial_dir.mkdir(exist_ok=True)
    filter_summary.to_csv(output_dir / "filter_summary.csv", index=False, encoding="utf-8-sig")
    outlier_summary.to_csv(output_dir / "outlier_filter_summary.csv", index=False, encoding="utf-8-sig")
    raw_overall.to_csv(output_dir / "overall_metrics_raw.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    data.nlargest(100, "tec1")[[
        "source_file", "time", "lat", "lon", "region", "tec0", "tec1",
        "gim_vtec", "cosmic_total_tec",
    ]].to_csv(output_dir / "largest_tec1_values.csv", index=False, encoding="utf-8-sig")
    write_method_note(
        output_dir / "method.txt", landmask, grid_size, min_bin_count,
        smooth_sigma, outlier_fraction,
    )
    for comparison in COMPARISONS:
        plot_scatter_by_region(
            comparison_data, overall, comparison, outlier_fraction,
            figures_dir / f"scatter_{comparison.key}.png", dpi,
        )
    plot_daily_metrics(daily, figures_dir / "daily_metrics.png", dpi)
    for region in REGION_ORDER:
        for comparison in COMPARISONS:
            spatial = spatial_outputs[(region, comparison.key)]
            spatial.to_csv(
                spatial_dir / f"spatial_metrics_{comparison.key}_{region}.csv",
                index=False, encoding="utf-8-sig",
            )
            plot_spatial_metrics(
                spatial, comparison, region, landmask, grid_size, min_bin_count,
                smooth_sigma, figures_dir / f"spatial_{comparison.key}_{region}.png", dpi,
            )


def parse_args() -> argparse.Namespace:
    """定义并解析分析所需的命令行参数。"""

    parser = argparse.ArgumentParser(description="COSMIC TEC/GIM 总体、陆地和海洋关系分析")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--landmask", type=Path, default=DEFAULT_LANDMASK)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--start-doy", type=int, default=61)
    parser.add_argument("--end-doy", type=int, default=91)
    parser.add_argument("--land-threshold", type=float, default=0.5)
    parser.add_argument("--grid-size", type=float, default=2.5, help="空间网格大小（度）")
    parser.add_argument("--min-bin-count", type=int, default=10, help="网格统计的最少样本数")
    parser.add_argument(
        "--smooth-sigma", type=float, default=1.2,
        help="空间图高斯平滑强度（网格数）；CSV 不平滑",
    )
    parser.add_argument(
        "--outlier-fraction", type=float, default=0.001,
        help="每个关系和区域按绝对残差删除的最高比例",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    """执行数据发现、清洗、区域统计、空间分析和结果输出。"""

    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{args.input_dir}")
    if not args.landmask.is_file():
        raise SystemExit(f"陆海掩膜不存在：{args.landmask}")
    if not 1 <= args.start_doy <= args.end_doy <= 366:
        raise SystemExit("DOY 范围必须满足 1 <= start-doy <= end-doy <= 366")
    if args.grid_size <= 0.0 or 360.0 % args.grid_size != 0.0 or 180.0 % args.grid_size != 0.0:
        raise SystemExit("grid-size 必须为能够整除 180 和 360 的正数")
    if args.min_bin_count < 2:
        raise SystemExit("min-bin-count 至少为 2")
    if args.smooth_sigma < 0.0:
        raise SystemExit("smooth-sigma 不能为负数")
    if not 0.0 <= args.land_threshold <= 1.0:
        raise SystemExit("land-threshold 必须位于 [0, 1]")
    if not 0.0 <= args.outlier_fraction < 0.1:
        raise SystemExit("outlier-fraction 必须满足 0 <= value < 0.1")

    files = discover_files(args.input_dir, args.year, args.start_doy, args.end_doy)
    if not files:
        raise SystemExit(
            f"{args.input_dir} 中没有 DOY {args.start_doy:03d}–{args.end_doy:03d} 的科学 CSV"
        )
    missing_days = sorted(set(range(args.start_doy, args.end_doy + 1)) - {d for d, _ in files})
    if missing_days:
        print("警告：缺少 DOY " + ", ".join(f"{value:03d}" for value in missing_days))

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"DOY{args.start_doy:03d}_{args.end_doy:03d}"
    )
    landmask = load_landmask(args.landmask, args.land_threshold)
    print(f"读取 {len(files)} 个逐日 CSV")
    print(f"陆海掩膜：{landmask.source}，变量={landmask.variable}，陆地阈值={landmask.threshold:g}")
    data, filter_summary = load_and_filter(files, args.year, landmask)
    print(filter_summary.to_string(index=False))
    comparison_data, outlier_summary = filter_comparison_outliers(data, args.outlier_fraction)
    raw_overall = overall_statistics(data)
    overall = overall_statistics(data, comparison_data)
    daily = daily_statistics(comparison_data)
    spatial_outputs = spatial_statistics(comparison_data, args.grid_size, args.min_bin_count)
    write_outputs(
        data, comparison_data, filter_summary, outlier_summary, raw_overall, overall,
        daily, spatial_outputs, landmask, output_dir, args.grid_size,
        args.min_bin_count, args.smooth_sigma, args.outlier_fraction, args.dpi,
    )
    print("\n过滤后的总体/陆地/海洋统计")
    print(overall[["region", "comparison", "count", "r", "bias", "rmse"]].to_string(index=False))
    print(f"\n结果目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
