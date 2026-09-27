"""验证 COSMIC 建模数据中的 GIM VTEC 插值和经纬度处理。

验证分为两部分：

1. 直接读取原始 IONEX，对随机网格节点、经度 ±360° 周期、±180° 接缝、
   时间中点线性插值和南北纬网格节点进行数值测试。
2. 汇总全年 COSMIC CSV，输出 residual 的分布、经纬度分箱、全球空间统计、
   逐日变化、地方时关系、GIM/tec0 散点以及极端样本清单。

脚本只读取科学 CSV 和原始 GIM，不修改建模数据。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from helper.interpolator import GIMInterpolator  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "modeling_csv_qc"
DEFAULT_GIM_DIR = PROJECT_ROOT / "Data" / "GIM2021" / "IGS"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cosmic2021_GIM_validation"
SCIENCE_PATTERN = re.compile(r"^cosmic2_model_(\d{4})_(\d{3})\.csv$")
REQUIRED_COLUMNS = (
    "datetime",
    "lat",
    "lon",
    "tec0",
    "gim_vtec",
    "residual",
    "local_time_s",
    "local_time_c",
    "is_land",
)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    """原子写出验证表格。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.12g")
    os.replace(temporary, destination)


def discover_science_files(input_dir: Path, year: int) -> list[tuple[int, Path]]:
    """发现并检查指定年份的逐日科学 CSV。"""

    selected: list[tuple[int, Path]] = []
    for path in input_dir.glob(f"cosmic2_model_{year}_???.csv"):
        match = SCIENCE_PATTERN.match(path.name)
        if match:
            selected.append((int(match.group(2)), path))
    selected.sort()
    actual = {doy for doy, _ in selected}
    missing = sorted(set(range(1, 366)).difference(actual))
    if missing:
        raise ValueError(f"缺少年积日科学 CSV：{missing}")
    return selected


def load_science_data(files: list[tuple[int, Path]]) -> pd.DataFrame:
    """合并全年验证所需字段，并保留源年积日和源行号。"""

    parts: list[pd.DataFrame] = []
    for index, (doy, path) in enumerate(files, start=1):
        frame = pd.read_csv(path, usecols=lambda name: name in REQUIRED_COLUMNS)
        missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
        frame["source_doy"] = doy
        frame["source_row"] = np.arange(len(frame), dtype=np.int32)
        parts.append(frame)
        if index % 50 == 0 or index == len(files):
            print(f"已读取 {index}/{len(files)} 个科学 CSV")

    data = pd.concat(parts, ignore_index=True)
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    numeric = [name for name in REQUIRED_COLUMNS if name != "datetime"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = data["datetime"].notna() & np.isfinite(data[numeric].to_numpy()).all(axis=1)
    if not valid.all():
        raise ValueError(f"科学 CSV 中有 {(~valid).sum()} 条验证必需字段无效")
    return data


def _append_test(
    rows: list[dict[str, object]],
    *,
    test_type: str,
    time: datetime,
    latitude: float,
    longitude: float,
    expected: float,
    actual: float,
    tolerance: float,
    detail: str = "",
) -> None:
    """记录一条插值数值测试及其是否通过。"""

    error = abs(float(actual) - float(expected))
    rows.append(
        {
            "test_type": test_type,
            "datetime": time.isoformat() + "Z",
            "lat": latitude,
            "lon": longitude,
            "expected": expected,
            "actual": actual,
            "abs_error": error,
            "tolerance": tolerance,
            "passed": bool(np.isfinite(error) and error <= tolerance),
            "detail": detail,
        }
    )


def run_algorithm_tests(
    gim: GIMInterpolator,
    year: int,
    test_days: int,
    tests_per_day: int,
    tolerance: float,
    seed: int,
) -> pd.DataFrame:
    """对全年分散日期执行原始 IONEX 节点、周期和时间插值测试。"""

    rng = np.random.default_rng(seed)
    sampled_doys = np.unique(
        np.rint(np.linspace(1, 365, max(2, test_days))).astype(int)
    )
    rows: list[dict[str, object]] = []

    for doy in sampled_doys:
        day = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
        raw = gim.load_day(day)
        times = list(raw["times"])
        lats = np.asarray(raw["lats"], dtype=float)
        lons = np.asarray(raw["lons"], dtype=float)
        cube = np.asarray(raw["tec"], dtype=float)
        if len(times) < 2:
            raise ValueError(f"DOY {doy:03d} 的 IONEX 时间层不足")

        # 最后一层通常是次日 00:00；避免查询时转而加载次日文件。
        for _ in range(tests_per_day):
            time_index = int(rng.integers(0, len(times) - 1))
            lat_index = int(rng.integers(0, len(lats)))
            lon_index = int(rng.integers(0, len(lons)))
            time = times[time_index]
            lat = float(lats[lat_index])
            lon = float(lons[lon_index])
            expected = float(cube[time_index, lat_index, lon_index])
            actual = gim.query(time, lat, lon)
            _append_test(
                rows,
                test_type="exact_ionex_node",
                time=time,
                latitude=lat,
                longitude=lon,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                detail=f"DOY={doy:03d}; t={time_index}; y={lat_index}; x={lon_index}",
            )

            # 同一物理经度用三种表示查询，应得到完全一致的结果。
            random_lon = float(rng.uniform(-180.0, 180.0))
            random_lat = float(rng.uniform(max(-87.5, lats.min()), min(87.5, lats.max())))
            reference = gim.query(time, random_lat, random_lon)
            for shift in (-360.0, 360.0):
                shifted = gim.query(time, random_lat, random_lon + shift)
                _append_test(
                    rows,
                    test_type="longitude_periodicity",
                    time=time,
                    latitude=random_lat,
                    longitude=random_lon + shift,
                    expected=reference,
                    actual=shifted,
                    tolerance=tolerance,
                    detail=f"base_lon={random_lon:.9g}; shift={shift:+.0f}",
                )

            # 两组等价的接缝经度分别从东西两侧跨越 ±180°。
            for west, east in ((-179.9, 180.1), (179.9, -180.1)):
                expected_seam = gim.query(time, random_lat, west)
                actual_seam = gim.query(time, random_lat, east)
                _append_test(
                    rows,
                    test_type="longitude_seam",
                    time=time,
                    latitude=random_lat,
                    longitude=east,
                    expected=expected_seam,
                    actual=actual_seam,
                    tolerance=tolerance,
                    detail=f"equivalent_lon={west:g}",
                )

            # 时间中点值应等于两端空间插值值的算术平均。
            interval_index = int(rng.integers(0, len(times) - 1))
            t1, t2 = times[interval_index], times[interval_index + 1]
            midpoint = t1 + (t2 - t1) / 2
            normalized_lon = gim._normalize_lon(random_lon, lons)
            v1 = gim._spatial_interp(
                lats, lons, cube[interval_index], random_lat, normalized_lon
            )
            v2 = gim._spatial_interp(
                lats, lons, cube[interval_index + 1], random_lat, normalized_lon
            )
            midpoint_value = gim.query(midpoint, random_lat, random_lon)
            _append_test(
                rows,
                test_type="time_midpoint",
                time=midpoint,
                latitude=random_lat,
                longitude=random_lon,
                expected=(v1 + v2) / 2.0,
                actual=midpoint_value,
                tolerance=tolerance,
                detail=f"t1={t1.isoformat()}; t2={t2.isoformat()}",
            )

        # 单独从北半球与南半球各取一个原始节点，检查纬度轴没有翻转。
        for hemisphere, candidates in (
            ("north", np.flatnonzero(lats > 20.0)),
            ("south", np.flatnonzero(lats < -20.0)),
        ):
            if not len(candidates):
                continue
            lat_index = int(rng.choice(candidates))
            lon_index = int(rng.integers(0, len(lons)))
            time_index = int(rng.integers(0, len(times) - 1))
            expected = float(cube[time_index, lat_index, lon_index])
            actual = gim.query(
                times[time_index], float(lats[lat_index]), float(lons[lon_index])
            )
            _append_test(
                rows,
                test_type="latitude_orientation",
                time=times[time_index],
                latitude=float(lats[lat_index]),
                longitude=float(lons[lon_index]),
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                detail=hemisphere,
            )

    return pd.DataFrame(rows)


def binned_statistics(data: pd.DataFrame, column: str, step: float) -> pd.DataFrame:
    """按经度、纬度或地方时分箱，计算 residual 稳健统计。"""

    if column == "lon":
        lower, upper = -180.0, 180.0
    elif column == "lat":
        lower, upper = -90.0, 90.0
    elif column == "local_time_hour":
        lower, upper = 0.0, 24.0
    else:
        raise ValueError(column)
    edges = np.arange(lower, upper + step, step)
    values = data[column].clip(lower=lower, upper=np.nextafter(upper, lower))
    bins = pd.cut(values, edges, right=False, include_lowest=True)
    grouped = data.groupby(bins, observed=True)["residual"]
    result = grouped.agg(count="count", mean="mean", median="median", std="std")
    result["p05"] = grouped.quantile(0.05)
    result["p95"] = grouped.quantile(0.95)
    centers = np.asarray([interval.mid for interval in result.index], dtype=float)
    result = result.reset_index(drop=True)
    result.insert(0, f"{column}_center", centers)
    return result


def spatial_statistics(data: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """将全年数据聚合为规则经纬度网格。"""

    work = data[["lat", "lon", "residual", "gim_vtec"]].copy()
    work["lat_center"] = (
        np.floor((work["lat"] + 90.0) / grid_size) * grid_size
        - 90.0
        + grid_size / 2.0
    ).clip(-90.0 + grid_size / 2.0, 90.0 - grid_size / 2.0)
    work["lon_center"] = (
        np.floor((work["lon"] + 180.0) / grid_size) * grid_size
        - 180.0
        + grid_size / 2.0
    ).clip(-180.0 + grid_size / 2.0, 180.0 - grid_size / 2.0)
    work["residual_squared"] = work["residual"] ** 2
    result = (
        work.groupby(["lat_center", "lon_center"], as_index=False)
        .agg(
            count=("residual", "size"),
            residual_mean=("residual", "mean"),
            residual_median=("residual", "median"),
            residual_squared_mean=("residual_squared", "mean"),
            gim_vtec_mean=("gim_vtec", "mean"),
        )
    )
    result["residual_rmse"] = np.sqrt(result.pop("residual_squared_mean"))
    return result


def daily_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """计算逐日 residual 和 GIM/tec0 相关性。"""

    work = data.copy()
    work["date"] = work["datetime"].dt.strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", sort=True):
        rows.append(
            {
                "date": date,
                "count": len(group),
                "residual_mean": group["residual"].mean(),
                "residual_median": group["residual"].median(),
                "residual_p05": group["residual"].quantile(0.05),
                "residual_p95": group["residual"].quantile(0.95),
                "residual_rmse": np.sqrt(np.mean(group["residual"] ** 2)),
                "gim_tec0_correlation": group[["gim_vtec", "tec0"]].corr().iloc[0, 1],
            }
        )
    return pd.DataFrame(rows)


def plot_distribution(data: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制 residual 分布、GIM/tec0 密度散点和经纬度分箱曲线。"""

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    lower, upper = data["residual"].quantile([0.001, 0.999])
    central_residual = data.loc[data["residual"].between(lower, upper), "residual"]
    axes[0, 0].hist(
        central_residual, bins=120, color="#2878b5", alpha=0.85
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(
        title="Residual distribution (central 99.8%)",
        xlabel="GIM VTEC − COSMIC tec0 (TECU)",
        ylabel="Count (log scale)",
    )

    plot_limit = float(
        max(data["tec0"].quantile(0.999), data["gim_vtec"].quantile(0.999))
    )
    sample = data.sample(min(300_000, len(data)), random_state=2021)
    density = axes[0, 1].hexbin(
        sample["tec0"],
        sample["gim_vtec"],
        gridsize=120,
        bins="log",
        mincnt=1,
        cmap="viridis",
        extent=(0, plot_limit, 0, plot_limit),
    )
    axes[0, 1].plot([0, plot_limit], [0, plot_limit], "r--", linewidth=1)
    axes[0, 1].set(
        xlim=(0, plot_limit),
        ylim=(0, plot_limit),
        title="GIM VTEC vs COSMIC tec0",
        xlabel="COSMIC tec0 (TECU)",
        ylabel="GIM VTEC (TECU)",
    )
    figure.colorbar(density, ax=axes[0, 1], label="log10(count)")

    lon_stats = binned_statistics(data, "lon", 5.0)
    lat_stats = binned_statistics(data, "lat", 5.0)
    for axis, stats, coordinate, label in (
        (axes[1, 0], lon_stats, "lon_center", "Longitude (deg)"),
        (axes[1, 1], lat_stats, "lat_center", "Latitude (deg)"),
    ):
        axis.fill_between(
            stats[coordinate], stats["p05"], stats["p95"], alpha=0.18, color="#2878b5"
        )
        axis.plot(stats[coordinate], stats["median"], color="#2878b5", label="Median")
        axis.plot(stats[coordinate], stats["mean"], color="#d65f32", label="Mean")
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(xlabel=label, ylabel="Residual (TECU)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 0].axvline(-180, color="red", linestyle="--", linewidth=0.8)
    axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1, 0].axvline(180, color="red", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Residual by longitude; red lines mark coordinate seams")
    axes[1, 1].set_title("Residual by latitude")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def _grid_array(
    spatial: pd.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把空间统计长表转换为二维网格。"""

    pivot = spatial.pivot(index="lat_center", columns="lon_center", values=value)
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def plot_spatial(spatial: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制 GIM 均值、residual 中位数、RMSE 和样本数全球分布。"""

    figure, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    specifications = (
        ("gim_vtec_mean", "Mean GIM VTEC", "viridis"),
        ("residual_median", "Median residual", "coolwarm"),
        ("residual_rmse", "Residual RMSE", "magma"),
        ("count", "Sample count", "cividis"),
    )
    for axis, (column, title, cmap) in zip(axes.flat, specifications):
        lon, lat, values = _grid_array(spatial, column)
        if column == "residual_median":
            robust = np.nanpercentile(np.abs(values), 98)
            vmin, vmax = -robust, robust
        else:
            vmin, vmax = None, np.nanpercentile(values, 99)
        image = axis.pcolormesh(
            lon, lat, values, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axis.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.88)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_temporal_and_local_time(
    data: pd.DataFrame, daily: pd.DataFrame, output: Path, dpi: int
) -> pd.DataFrame:
    """绘制逐日和地方时 residual，并返回地方时分箱统计。"""

    local_angle = np.arctan2(
        data["local_time_s"].to_numpy(dtype=float),
        data["local_time_c"].to_numpy(dtype=float),
    )
    data = data.copy()
    data["local_time_hour"] = np.mod(local_angle, 2.0 * np.pi) * 24.0 / (2.0 * np.pi)
    local = binned_statistics(data, "local_time_hour", 1.0)

    dates = pd.to_datetime(daily["date"])
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].fill_between(
        dates, daily["residual_p05"], daily["residual_p95"], alpha=0.18
    )
    axes[0].plot(dates, daily["residual_median"], label="Median", linewidth=1)
    axes[0].plot(dates, daily["residual_mean"], label="Mean", linewidth=1)
    axes[0].set(ylabel="Residual (TECU)", title="Daily residual statistics")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].fill_between(
        local["local_time_hour_center"], local["p05"], local["p95"], alpha=0.18
    )
    axes[1].plot(local["local_time_hour_center"], local["median"], label="Median")
    axes[1].plot(local["local_time_hour_center"], local["mean"], label="Mean")
    axes[1].set(
        xlim=(0, 24),
        xticks=np.arange(0, 25, 3),
        xlabel="Local solar time (hour)",
        ylabel="Residual (TECU)",
        title="Residual by local solar time",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return local


def build_summary(
    data: pd.DataFrame,
    tests: pd.DataFrame,
    lon_stats: pd.DataFrame,
    residual_tolerance: float,
) -> pd.DataFrame:
    """汇总算法测试、经度接缝和全年数据的关键验证指标。"""

    residual_recomputed = data["gim_vtec"] - data["tec0"]
    residual_error = np.abs(data["residual"] - residual_recomputed)
    west = data.loc[data["lon"].between(-180.0, -175.0, inclusive="left"), "residual"]
    east = data.loc[data["lon"].between(175.0, 180.0, inclusive="both"), "residual"]
    zero_west = data.loc[data["lon"].between(-5.0, 0.0, inclusive="left"), "residual"]
    zero_east = data.loc[data["lon"].between(0.0, 5.0, inclusive="left"), "residual"]
    adjacent_jump = np.abs(np.diff(lon_stats["median"].to_numpy(dtype=float)))
    correlation = data[["gim_vtec", "tec0"]].corr().iloc[0, 1]

    items = [
        ("records", len(data), "count"),
        ("algorithm_tests", len(tests), "count"),
        ("algorithm_test_failures", int((~tests["passed"]).sum()), "count"),
        ("maximum_algorithm_abs_error", tests["abs_error"].max(), "TECU"),
        ("maximum_saved_residual_error", residual_error.max(), "TECU"),
        ("residual_formula_within_tolerance", bool((residual_error <= residual_tolerance).all()), "bool"),
        ("gim_tec0_correlation", correlation, "correlation"),
        ("residual_mean", data["residual"].mean(), "TECU"),
        ("residual_median", data["residual"].median(), "TECU"),
        ("residual_std", data["residual"].std(), "TECU"),
        ("residual_p01", data["residual"].quantile(0.01), "TECU"),
        ("residual_p99", data["residual"].quantile(0.99), "TECU"),
        ("residual_negative_count", int((data["residual"] < 0).sum()), "count"),
        ("residual_below_minus20_count", int((data["residual"] < -20).sum()), "count"),
        ("gim_vtec_min", data["gim_vtec"].min(), "TECU"),
        ("gim_vtec_max", data["gim_vtec"].max(), "TECU"),
        ("seam_west_median", west.median(), "TECU"),
        ("seam_east_median", east.median(), "TECU"),
        ("seam_median_difference", abs(west.median() - east.median()), "TECU"),
        ("zero_west_median", zero_west.median(), "TECU"),
        ("zero_east_median", zero_east.median(), "TECU"),
        ("zero_median_difference", abs(zero_west.median() - zero_east.median()), "TECU"),
        ("maximum_adjacent_5deg_median_jump", np.nanmax(adjacent_jump), "TECU"),
        ("median_adjacent_5deg_median_jump", np.nanmedian(adjacent_jump), "TECU"),
    ]
    return pd.DataFrame(items, columns=["metric", "value", "unit"])


def write_method_note(output: Path, args: argparse.Namespace) -> None:
    """写出结果解释和异常判别原则。"""

    note = f"""# COSMIC/GIM 插值验证说明

## 算法级验证

- 从 {args.test_days} 个全年均匀分布的日期抽取 IONEX。
- 每日执行 {args.tests_per_day} 组原始网格节点测试。
- 同时测试经度 ±360° 等价、±180° 接缝、相邻时间层中点和南北纬节点。
- 单项绝对误差容限：{args.algorithm_tolerance:g} TECU。

算法级测试直接比较原始 IONEX 数值，优先级高于 residual 分布。只要节点、
经度周期和时间中点测试全部通过，就能排除主要的经度范围、纬度倒置和时间
线性插值实现错误。

## 数据级诊断

- residual 定义：GIM VTEC − COSMIC tec0。
- residual 的非零均值不等同于插值错误，因为两种产品的观测范围不同。
- 重点检查 ±180°、0°附近是否出现相对相邻经度箱异常放大的跳变。
- 全球地图上的整列断裂通常提示经度接缝错误；南北结构整体镜像通常提示
  纬度轴倒置；固定 UTC 时刻的周期跳变可能提示时间层或日期选择错误。
- 极端 residual 需回查原始 ionPrf，不能仅凭统计阈值判定为坏数据。
"""
    output.write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """定义并解析验证参数。"""

    parser = argparse.ArgumentParser(description="验证 COSMIC 数据中的 GIM 插值")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--gim-dir", type=Path, default=DEFAULT_GIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--test-days", type=int, default=12)
    parser.add_argument("--tests-per-day", type=int, default=12)
    parser.add_argument("--algorithm-tolerance", type=float, default=1e-9)
    parser.add_argument("--residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--grid-size", type=float, default=5.0)
    parser.add_argument("--extreme-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    """运行算法和数据两级验证并写出全部表格、图件和结论。"""

    args = parse_args()
    for path, label in ((args.input_dir, "科学 CSV 目录"), (args.gim_dir, "GIM 目录")):
        if not path.is_dir():
            raise SystemExit(f"{label}不存在：{path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_science_files(args.input_dir, args.year)
    gim = GIMInterpolator(args.gim_dir)
    print("开始执行原始 IONEX 插值算法测试")
    tests = run_algorithm_tests(
        gim,
        args.year,
        args.test_days,
        args.tests_per_day,
        args.algorithm_tolerance,
        args.seed,
    )
    atomic_csv(tests, args.output_dir / "algorithm_tests.csv")

    print("开始读取全年 COSMIC 建模数据")
    data = load_science_data(files)
    lon_stats = binned_statistics(data, "lon", 5.0)
    lat_stats = binned_statistics(data, "lat", 5.0)
    spatial = spatial_statistics(data, args.grid_size)
    daily = daily_statistics(data)
    residual_recomputed = data["gim_vtec"] - data["tec0"]
    data["residual_formula_error"] = data["residual"] - residual_recomputed
    extremes = data.reindex(data["residual"].abs().nlargest(args.extreme_count).index)
    extremes = extremes.sort_values("residual", key=lambda series: series.abs(), ascending=False)

    atomic_csv(lon_stats, args.output_dir / "longitude_bin_statistics.csv")
    atomic_csv(lat_stats, args.output_dir / "latitude_bin_statistics.csv")
    atomic_csv(spatial, args.output_dir / "spatial_grid_statistics.csv")
    atomic_csv(daily, args.output_dir / "daily_statistics.csv")
    atomic_csv(extremes, args.output_dir / "extreme_residual_samples.csv")

    plot_distribution(data, args.output_dir / "residual_diagnostics.png", args.dpi)
    plot_spatial(spatial, args.output_dir / "spatial_diagnostics.png", args.dpi)
    local = plot_temporal_and_local_time(
        data, daily, args.output_dir / "temporal_local_time_diagnostics.png", args.dpi
    )
    atomic_csv(local, args.output_dir / "local_time_statistics.csv")

    summary = build_summary(
        data, tests, lon_stats, residual_tolerance=args.residual_tolerance
    )
    atomic_csv(summary, args.output_dir / "validation_summary.csv")
    write_method_note(args.output_dir / "README_validation.md", args)

    failures = int((~tests["passed"]).sum())
    print(summary.to_string(index=False))
    print(f"验证完成：算法测试 {len(tests)} 项，失败 {failures} 项")
    print(f"输出目录：{args.output_dir.resolve()}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
