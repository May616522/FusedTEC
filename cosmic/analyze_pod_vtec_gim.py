"""按下垫面类型检验 COSMIC tec0 加 POD 顶部 VTEC 与 GIM 的一致性。

脚本既可读取单个处理后 CSV，也可批量读取多日目录；分别分析全球、陆地和
海洋子集，并导出全时段与逐日指标。偏差统一定义为 ``GIM-COSMIC估计值``。
默认对每种方法和区域剔除绝对残差最大的 0.1%，同时另行保留原始指标。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import analyze_cosmic_tec_gim as shared


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "Data"
    / "Cosmic2021"
    / "processed_csv_gim_pod"
    / "cosmic2_ionPrF_2021_075_pod.csv"
)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim_pod"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "Cosmic2_POD_GIM_analysis"
DEFAULT_LANDMASK = PROJECT_ROOT / "Data" / "Other" / "landmask_static.nc"
REGIONS = ("all", "land", "ocean")
REGION_LABELS = {"all": "Overall", "land": "Land", "ocean": "Ocean"}


@dataclass(frozen=True)
class Method:
    """描述一种顶部 TEC 补偿方案及其使用的数据列。"""

    key: str
    label: str
    top_column: str | None


METHODS = (
    Method("tec0_only", "tec0 only", None),
    Method("tec0_plus_tec1", "tec0 + CDAAC tec1", "tec1"),
    Method("tec0_plus_pod_fk", "tec0 + POD (STEC/M)", "pod_vtec_fk"),
    Method(
        "tec0_plus_pod_eq2",
        "tec0 + POD (paper Eq. 2 literal)",
        "pod_vtec_eq2_literal",
    ),
)


def load_data(path: Path, landmask: shared.LandMask) -> pd.DataFrame:
    """读取单个 POD 增强 CSV，清洗字段并添加日期与陆海分类。"""

    frame = pd.read_csv(path)
    required = {
        "time",
        "lat",
        "lon",
        "tec0",
        "tec1",
        "pod_vtec_fk",
        "pod_vtec_eq2_literal",
        "gim_vtec",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
    numeric_columns = sorted(required - {"time"})
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["datetime"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame["lon"] = (frame["lon"] + 180.0) % 360.0 - 180.0
    base_valid = (
        frame["datetime"].notna()
        & np.isfinite(frame[["lat", "lon", "tec0", "gim_vtec"]].to_numpy()).all(axis=1)
        & frame["lat"].between(-90.0, 90.0)
    )
    frame = frame.loc[base_valid].copy()
    frame["source_file"] = path.name
    frame["observation_date"] = frame["datetime"].dt.strftime("%Y-%m-%d")
    frame["observation_doy"] = frame["datetime"].dt.dayofyear.astype(int)
    file_date = re.search(r"_(\d{4})_(\d{3})_pod\.csv$", path.name)
    if file_date:
        source_year, source_doy = map(int, file_date.groups())
        source_date = pd.to_datetime(f"{source_year}-{source_doy:03d}", format="%Y-%j")
        frame["date"] = source_date.strftime("%Y-%m-%d")
        frame["doy"] = source_doy
    else:
        frame["date"] = frame["observation_date"]
        frame["doy"] = frame["observation_doy"]
    is_land = landmask.is_land(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    frame["region"] = np.where(is_land, "land", "ocean")
    frame["tec0_only"] = frame["tec0"]
    frame["tec0_plus_tec1"] = frame["tec0"] + frame["tec1"]
    invalid_tec1 = np.isclose(frame["tec1"], -999.0, rtol=0.0, atol=1.0e-9)
    frame.loc[invalid_tec1, "tec0_plus_tec1"] = np.nan
    frame["tec0_plus_pod_fk"] = frame["tec0"] + frame["pod_vtec_fk"]
    frame["tec0_plus_pod_eq2"] = frame["tec0"] + frame["pod_vtec_eq2_literal"]
    return frame.reset_index(drop=True)


def discover_inputs(input_file: Path | None, input_dir: Path, input_glob: str, doys: set[int] | None) -> list[Path]:
    """确定单文件或目录批处理模式下需要分析的输入文件。"""

    if input_file is not None:
        candidates = [input_file]
    else:
        candidates = sorted(input_dir.glob(input_glob))
    if doys is not None:
        selected: list[Path] = []
        for path in candidates:
            match = re.search(r"_(\d{3})_pod\.csv$", path.name)
            if match and int(match.group(1)) in doys:
                selected.append(path)
        candidates = selected
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"输入CSV不存在：{missing[0]}")
    if not candidates:
        raise FileNotFoundError(f"未找到输入CSV：{input_dir / input_glob}")
    return candidates


def region_data(data: pd.DataFrame, region: str) -> pd.DataFrame:
    """返回全球、陆地或海洋区域对应的数据子集。"""

    return data if region == "all" else data.loc[data["region"] == region]


def calculate_metrics(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    """计算两组 TEC 的样本数、偏差、误差、相关性和回归参数。"""

    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[valid], y_values[valid]
    count = len(x_values)
    if count == 0:
        return {"count": 0, "r": np.nan, "bias": np.nan, "rmse": np.nan}
    residual = y_values - x_values
    correlation = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if count >= 2 and np.std(x_values) > 0 and np.std(y_values) > 0
        else np.nan
    )
    return {
        "count": count,
        "r": correlation,
        "bias": float(np.mean(residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


# 对指定方法和区域计算原始指标，再按残差阈值剔除异常值并重新统计。
def filter_and_metrics(
    data: pd.DataFrame, outlier_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    raw_rows: list[dict[str, object]] = []
    robust_rows: list[dict[str, object]] = []
    filtered: dict[tuple[str, str], pd.DataFrame] = {}
    for region in REGIONS:
        regional = region_data(data, region)
        for method in METHODS:
            valid = np.isfinite(regional[[method.key, "gim_vtec"]].to_numpy()).all(axis=1)
            selected = regional.loc[valid].copy()
            raw_metric = calculate_metrics(selected[method.key], selected["gim_vtec"])
            raw_rows.append(
                {
                    "region": region,
                    "method": method.key,
                    "label": method.label,
                    "bias_definition": f"gim_vtec - {method.key}",
                    **raw_metric,
                }
            )
            residual = (selected["gim_vtec"] - selected[method.key]).abs()
            if outlier_fraction == 0.0 or selected.empty:
                cutoff = float("inf")
                robust = selected
            else:
                cutoff = float(residual.quantile(1.0 - outlier_fraction))
                robust = selected.loc[residual <= cutoff].copy()
            filtered[(region, method.key)] = robust
            robust_metric = calculate_metrics(robust[method.key], robust["gim_vtec"])
            robust_rows.append(
                {
                    "region": region,
                    "method": method.key,
                    "label": method.label,
                    "bias_definition": f"gim_vtec - {method.key}",
                    "absolute_residual_cutoff_tecu": cutoff,
                    "removed_count": len(selected) - len(robust),
                    **robust_metric,
                }
            )
    return pd.DataFrame(raw_rows), pd.DataFrame(robust_rows), filtered


# 按观测日期重复各方法、各区域的筛选与指标计算。
def calculate_daily_metrics(
    data: pd.DataFrame, outlier_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    robust_rows: list[dict[str, object]] = []
    day_columns = data[["date", "doy"]].drop_duplicates().sort_values("doy")
    for day in day_columns.itertuples(index=False):
        daily = data.loc[data["doy"] == day.doy]
        for region in REGIONS:
            regional = region_data(daily, region)
            for method in METHODS:
                valid = np.isfinite(regional[[method.key, "gim_vtec"]].to_numpy()).all(axis=1)
                selected = regional.loc[valid].copy()
                raw_metric = calculate_metrics(selected[method.key], selected["gim_vtec"])
                base = {
                    "date": day.date,
                    "doy": int(day.doy),
                    "region": region,
                    "method": method.key,
                    "label": method.label,
                    "bias_definition": f"gim_vtec - {method.key}",
                }
                raw_rows.append({**base, **raw_metric})
                residual = (selected["gim_vtec"] - selected[method.key]).abs()
                if outlier_fraction == 0.0 or selected.empty:
                    cutoff = float("inf")
                    robust = selected
                else:
                    cutoff = float(residual.quantile(1.0 - outlier_fraction))
                    robust = selected.loc[residual <= cutoff]
                robust_metric = calculate_metrics(robust[method.key], robust["gim_vtec"])
                robust_rows.append(
                    {
                        **base,
                        "absolute_residual_cutoff_tecu": cutoff,
                        "removed_count": len(selected) - len(robust),
                        **robust_metric,
                    }
                )
    return pd.DataFrame(raw_rows), pd.DataFrame(robust_rows)


def _metric_text(row: pd.Series) -> str:
    return (
        f"N = {int(row['count']):,}\n"
        f"R = {row['r']:.4f}\n"
        f"Bias = {row['bias']:.4f} TECU\n"
        f"RMSE = {row['rmse']:.4f} TECU"
    )


# 为每种补偿方法绘制 COSMIC 估计值与 GIM 的分区域散点图。
def plot_scatter(
    filtered: dict[tuple[str, str], pd.DataFrame],
    metrics: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    key = "tec0_plus_pod_fk"
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
    for axis, region in zip(axes, REGIONS):
        subset = filtered[(region, key)]
        x = subset[key].to_numpy(dtype=float)
        y = subset["gim_vtec"].to_numpy(dtype=float)
        low, high = np.percentile(np.concatenate((x, y)), [0.5, 99.5])
        padding = max((high - low) * 0.04, 0.1)
        low, high = low - padding, high + padding
        density = axis.hexbin(
            x,
            y,
            gridsize=65,
            bins="log",
            mincnt=1,
            extent=(low, high, low, high),
            cmap="viridis",
        )
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1.0)
        axis.set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel="tec0 + POD VTEC (TECU)",
            ylabel="GIM VTEC (TECU)",
            title=REGION_LABELS[region],
        )
        axis.grid(True, alpha=0.22)
        row = metrics.loc[(metrics["region"] == region) & (metrics["method"] == key)].iloc[0]
        axis.text(
            0.03,
            0.97,
            _metric_text(row),
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.5"},
        )
        fig.colorbar(density, ax=axis, pad=0.02).set_label("log10 observation count")
    fig.suptitle("COSMIC tec0 + POD topside VTEC vs GIM VTEC\nBias = GIM - (tec0 + POD)")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """用柱状图比较各顶部 TEC 补偿方法的核心统计指标。"""

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    specs = (("r", "Pearson R"), ("bias", "Bias: GIM - estimate (TECU)"), ("rmse", "RMSE (TECU)"))
    colors = ["#777777", "#c44e52", "#2878b5", "#dd8452"]
    for row_index, region in enumerate(REGIONS):
        subset = metrics.loc[metrics["region"] == region].set_index("method").loc[[m.key for m in METHODS]]
        for column_index, (metric, label) in enumerate(specs):
            axis = axes[row_index, column_index]
            values = subset[metric].to_numpy(dtype=float)
            axis.bar(np.arange(len(METHODS)), values, color=colors)
            axis.set_xticks(np.arange(len(METHODS)), ["tec0", "tec0+tec1", "tec0+POD\nSTEC/M", "tec0+POD\nEq.(2)"], rotation=15)
            axis.set_ylabel(label)
            axis.set_title(REGION_LABELS[region])
            axis.grid(True, axis="y", alpha=0.25)
            if metric == "bias":
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            for index, value in enumerate(values):
                axis.text(index, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Comparison of topside correction methods against GIM")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# 绘制不同方法的残差分布，直观检查偏差和长尾异常值。
def plot_residual_histograms(
    filtered: dict[tuple[str, str], pd.DataFrame], output: Path, dpi: int
) -> None:
    key = "tec0_plus_pod_fk"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    residuals = [
        filtered[(region, key)]["gim_vtec"] - filtered[(region, key)][key]
        for region in REGIONS
    ]
    limit = max(float(np.percentile(np.abs(values), 99.0)) for values in residuals)
    bins = np.linspace(-limit, limit, 60)
    for axis, region, residual in zip(axes, REGIONS, residuals):
        axis.hist(residual, bins=bins, color="#2878b5", alpha=0.82)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        axis.set(
            xlabel="GIM - (tec0 + POD) (TECU)",
            ylabel="Observations",
            title=REGION_LABELS[region],
        )
        axis.grid(True, alpha=0.22)
    fig.suptitle("Residual distributions after POD topside correction")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_daily_metrics(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制各方法和区域统计指标的逐日变化。"""

    key = "tec0_plus_pod_fk"
    subset = metrics.loc[metrics["method"] == key].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    specs = (("r", "Pearson R"), ("bias", "Bias: GIM - (tec0 + POD) (TECU)"), ("rmse", "RMSE (TECU)"))
    colors = {"all": "#222222", "land": "#2ca02c", "ocean": "#1f77b4"}
    markers = {"all": "o", "land": "^", "ocean": "s"}
    for axis, (column, ylabel) in zip(axes, specs):
        for region in REGIONS:
            regional = subset.loc[subset["region"] == region].sort_values("doy")
            axis.plot(
                regional["doy"],
                regional[column],
                marker=markers[region],
                color=colors[region],
                linewidth=1.5,
                markersize=5,
                label=REGION_LABELS[region],
            )
        if column == "bias":
            axis.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xlabel("Day of year (2021)")
    axes[-1].set_xticks(sorted(subset["doy"].unique()))
    fig.suptitle("Daily agreement of tec0 + POD topside VTEC with GIM")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_daily_counts(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制各区域每日参与统计的有效样本数。"""

    subset = metrics.loc[metrics["method"] == "tec0_plus_pod_fk"].copy()
    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    colors = {"all": "#222222", "land": "#2ca02c", "ocean": "#1f77b4"}
    for region in REGIONS:
        regional = subset.loc[subset["region"] == region].sort_values("doy")
        axis.plot(regional["doy"], regional["count"], marker="o", color=colors[region], label=REGION_LABELS[region])
    axis.set(xlabel="Day of year (2021)", ylabel="Valid observations", title="Daily sample coverage after outlier filtering")
    axis.set_xticks(sorted(subset["doy"].unique()))
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def spatial_grid(data: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """把残差按经纬度网格聚合为空间统计表。"""

    binned = data.copy()
    binned["lon_index"] = np.floor((binned["lon"] + 180.0) / grid_size).astype(int)
    binned["lat_index"] = np.floor((binned["lat"] + 90.0) / grid_size).astype(int)
    binned["residual"] = binned["gim_vtec"] - binned["tec0_plus_pod_fk"]
    grid = binned.groupby(["lat_index", "lon_index"], sort=True).agg(
        count=("residual", "size"),
        pod_total_median=("tec0_plus_pod_fk", "median"),
        gim_median=("gim_vtec", "median"),
        residual_median=("residual", "median"),
    ).reset_index()
    grid["lat_center"] = -90.0 + (grid["lat_index"] + 0.5) * grid_size
    grid["lon_center"] = -180.0 + (grid["lon_index"] + 0.5) * grid_size
    return grid


def _array_grid(grid: pd.DataFrame, column: str, grid_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.arange(-180.0 + grid_size / 2.0, 180.0, grid_size)
    lat = np.arange(-90.0 + grid_size / 2.0, 90.0, grid_size)
    values = np.full((len(lat), len(lon)), np.nan)
    values[grid["lat_index"].to_numpy(dtype=int), grid["lon_index"].to_numpy(dtype=int)] = grid[column].to_numpy(dtype=float)
    return lon, lat, values


def _smooth(values: np.ndarray, counts: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    weights = np.where(valid, counts, 0.0)
    numerator = gaussian_filter(np.where(valid, values * weights, 0.0), sigma, mode=("nearest", "wrap"))
    denominator = gaussian_filter(weights, sigma, mode=("nearest", "wrap"))
    support = gaussian_filter(valid.astype(float), sigma, mode=("nearest", "wrap"))
    result = np.full_like(values, np.nan)
    use = (denominator > 0) & (support >= 0.04)
    result[use] = numerator[use] / denominator[use]
    return result


# 绘制各顶部补偿方法偏差和 RMSE 的全球空间分布。
def plot_spatial(
    grid: pd.DataFrame,
    region: str,
    landmask: shared.LandMask,
    grid_size: float,
    sigma: float,
    output: Path,
    dpi: int,
) -> None:
    projection = {"projection": shared.ccrs.PlateCarree()} if shared.HAS_CARTOPY else {}
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), subplot_kw=projection, constrained_layout=True)
    lon, lat, counts = _array_grid(grid, "count", grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    region_mask = np.ones_like(lon_mesh, dtype=bool)
    if region != "all":
        is_land = landmask.is_land(lat_mesh.ravel(), lon_mesh.ravel()).reshape(lat_mesh.shape)
        region_mask = is_land if region == "land" else ~is_land
    specs = (
        ("count", "Observation count", "viridis"),
        ("pod_total_median", "tec0 + POD VTEC (TECU)", "turbo"),
        ("gim_median", "GIM VTEC (TECU)", "turbo"),
        ("residual_median", "GIM - (tec0 + POD) (TECU)", "RdBu_r"),
    )
    pod_grid = _array_grid(grid, "pod_total_median", grid_size)[2]
    gim_grid = _array_grid(grid, "gim_median", grid_size)[2]
    common_max = float(np.nanpercentile(np.concatenate((pod_grid[np.isfinite(pod_grid)], gim_grid[np.isfinite(gim_grid)])), 98.0))
    for axis, (column, title, cmap) in zip(axes.flat, specs):
        raw = _array_grid(grid, column, grid_size)[2]
        if column == "count":
            smoothing_weight = np.where(np.isfinite(raw), 1.0, 0.0)
        else:
            smoothing_weight = counts
        values = _smooth(raw, smoothing_weight, sigma)
        values[~region_mask] = np.nan
        if column in {"pod_total_median", "gim_median"}:
            levels = np.linspace(0.0, common_max, 21)
            extend = "max"
        elif column == "residual_median":
            limit = float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 98.0))
            levels = np.linspace(-limit, limit, 21)
            extend = "both"
        else:
            levels = np.linspace(0.0, float(np.nanpercentile(values[np.isfinite(values)], 99.0)), 21)
            extend = "max"
        kwargs: dict[str, object] = {"levels": levels, "cmap": cmap, "extend": extend}
        if shared.HAS_CARTOPY:
            kwargs["transform"] = shared.ccrs.PlateCarree()
        image = axis.contourf(lon, lat, values, **kwargs)
        if shared.HAS_CARTOPY:
            axis.set_extent([-180, 180, -45, 45], crs=shared.ccrs.PlateCarree())
            axis.coastlines(resolution="110m", linewidth=0.65, color="0.2")
            lines = axis.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
            lines.top_labels = False
            lines.right_labels = False
        else:
            axis.set(xlim=(-180, 180), ylim=(-45, 45), xlabel="Longitude", ylabel="Latitude")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, pad=0.025).set_label(title)
    fig.suptitle(f"{REGION_LABELS[region]} spatial comparison: tec0 + POD vs GIM")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""

    parser = argparse.ArgumentParser(description="tec0+POD与GIM的陆地/海洋相关性分析")
    parser.add_argument("--input", type=Path, default=None, help="单个POD校正CSV；指定后忽略input-dir")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-glob", default="cosmic2_ionPrF_*_pod.csv")
    parser.add_argument("--doys", default=None, help="仅分析指定DOY，例如061,065,067")
    parser.add_argument("--landmask", type=Path, default=DEFAULT_LANDMASK)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--land-threshold", type=float, default=0.5)
    parser.add_argument("--outlier-fraction", type=float, default=0.001)
    parser.add_argument("--grid-size", type=float, default=5.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    """完成数据加载、方法比较、逐日统计、空间分析和文件输出。"""

    args = parse_args()
    if not args.landmask.is_file():
        raise SystemExit(f"陆海掩膜不存在：{args.landmask}")
    if not 0 <= args.outlier_fraction < 0.1:
        raise SystemExit("outlier-fraction必须满足0 <= value < 0.1")
    try:
        selected_doys = None if args.doys is None else {int(value) for value in args.doys.split(",") if value.strip()}
    except ValueError as error:
        raise SystemExit("--doys必须是逗号分隔的整数，例如061,065,067") from error
    try:
        inputs = discover_inputs(args.input, args.input_dir, args.input_glob, selected_doys)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    landmask = shared.load_landmask(args.landmask, args.land_threshold)
    frames = [load_data(path, landmask) for path in inputs]
    data = pd.concat(frames, ignore_index=True)
    available_doys = sorted(data["doy"].unique())
    if args.output_dir is None:
        period_name = (
            f"DOY{available_doys[0]:03d}"
            if len(available_doys) == 1
            else f"DOY{available_doys[0]:03d}-{available_doys[-1]:03d}_{len(available_doys)}days"
        )
        output_dir = DEFAULT_OUTPUT_ROOT / period_name
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    spatial_csv = output_dir / "spatial_csv"
    figures.mkdir(exist_ok=True)
    spatial_csv.mkdir(exist_ok=True)

    raw, robust, filtered = filter_and_metrics(data, args.outlier_fraction)
    daily_raw, daily_robust = calculate_daily_metrics(data, args.outlier_fraction)
    raw.to_csv(output_dir / "overall_metrics_raw.csv", index=False, encoding="utf-8-sig")
    robust.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    daily_raw.to_csv(output_dir / "daily_metrics_raw.csv", index=False, encoding="utf-8-sig")
    daily_robust.to_csv(output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    manifest_data = data.assign(boundary_timestamp=data["observation_doy"] != data["doy"])
    manifest = (
        manifest_data.groupby(["source_file", "date", "doy"], as_index=False)
        .agg(
            input_rows=("source_file", "size"),
            observation_date_min=("observation_date", "min"),
            observation_date_max=("observation_date", "max"),
            boundary_timestamp_rows=("boundary_timestamp", "sum"),
        )
        .sort_values("doy")
    )
    manifest.to_csv(output_dir / "input_manifest.csv", index=False, encoding="utf-8-sig")
    plot_scatter(filtered, robust, figures / "scatter_tec0_plus_pod_vs_gim.png", args.dpi)
    plot_method_comparison(robust, figures / "method_comparison.png", args.dpi)
    plot_residual_histograms(filtered, figures / "residual_histograms.png", args.dpi)
    plot_daily_metrics(daily_robust, figures / "daily_metrics_tec0_plus_pod_vs_gim.png", args.dpi)
    plot_daily_counts(daily_robust, figures / "daily_sample_counts.png", args.dpi)
    for region in REGIONS:
        grid = spatial_grid(filtered[(region, "tec0_plus_pod_fk")], args.grid_size)
        grid.to_csv(spatial_csv / f"spatial_metrics_{region}.csv", index=False, encoding="utf-8-sig")
        plot_spatial(
            grid,
            region,
            landmask,
            args.grid_size,
            args.smooth_sigma,
            figures / f"spatial_tec0_plus_pod_vs_gim_{region}.png",
            args.dpi,
        )
    print(robust.loc[robust["method"] == "tec0_plus_pod_fk", ["region", "count", "r", "bias", "rmse"]].to_string(index=False))
    print(f"输入日期：{','.join(f'{value:03d}' for value in available_doys)}")
    print(f"结果目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
