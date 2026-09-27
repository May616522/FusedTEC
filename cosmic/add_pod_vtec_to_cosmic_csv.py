"""将 COSMIC-2 podTc2 的斜向 TEC（STEC）转换为顶部垂直 TEC 并写入掩星 CSV。

处理流程参考 Wu 等（2024）第 2.2 节：筛选高仰角 POD 观测，使用
Foelsche & Kirchengast 厚壳映射函数将 STEC 映射为 VTEC，在纬度—地方时
空间建立模型，然后把模型插值到每个无线电掩星事件的位置和时刻。

映射因子存在两种公式约定：Zhong 等（2016）定义 M=STEC/VTEC，因此标准
换算为 VTEC=STEC/M；Wu 等（2024）的公式（2）印作 VTEC=M*STEC。脚本同时
导出两套结果用于诊断，其中 ``pod_vtec`` 和 ``tec0_plus_pod_vtec`` 使用标准
除法。脚本始终另存结果，不覆盖原始科学数据 CSV。
"""

from __future__ import annotations

import argparse
import os
import re
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_POD_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "POD"
DEFAULT_CSV_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim_pod"
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT / "Results" / "Cosmic2_POD_VTEC"

GPS_MINUS_UTC_SECONDS_2021 = 18.0
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
MEAN_EARTH_RADIUS_KM = 6371.0
POD_ARCHIVE_PATTERN = re.compile(r"podTc2_.*_(\d{4})_(\d{3})\.tar\.gz$", re.IGNORECASE)


@dataclass(frozen=True)
class PodModel:
    """保存 POD 纬度—地方时网格及其最近邻查询结构。"""

    grid: pd.DataFrame
    tree: cKDTree
    features: np.ndarray


def _as_float(variable: netCDF4.Variable) -> np.ndarray:
    """把 NetCDF 变量展平为浮点数组，并将掩码值转换为 NaN。"""

    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _attr_int(dataset: netCDF4.Dataset, name: str, default: int = -999) -> int:
    """安全读取 NetCDF 整数属性，读取失败时返回默认值。"""

    try:
        return int(dataset.getncattr(name))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return default


def _attr_text(dataset: netCDF4.Dataset, name: str, default: str = "") -> str:
    """安全读取并规范化 NetCDF 文本属性。"""

    try:
        return str(dataset.getncattr(name)).strip().upper()
    except (AttributeError, TypeError, ValueError):
        return default


def _safe_member(member: tarfile.TarInfo) -> bool:
    """判断压缩包成员是否为安全的普通文件，防止路径穿越。"""

    path = PurePosixPath(member.name.replace("\\", "/"))
    return member.isfile() and not path.is_absolute() and ".." not in path.parts


def ecef_to_geodetic(
    x_km: np.ndarray, y_km: np.ndarray, z_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 WGS-84 地心地固坐标批量转换为大地纬度、经度和高度。"""

    eccentricity_squared = WGS84_F * (2.0 - WGS84_F)
    horizontal = np.hypot(x_km, y_km)
    longitude = np.degrees(np.arctan2(y_km, x_km))
    latitude = np.arctan2(z_km, horizontal * (1.0 - eccentricity_squared))
    altitude = np.zeros_like(latitude)
    for _ in range(7):
        sin_latitude = np.sin(latitude)
        prime_vertical = WGS84_A_KM / np.sqrt(
            1.0 - eccentricity_squared * sin_latitude**2
        )
        altitude = horizontal / np.cos(latitude) - prime_vertical
        latitude = np.arctan2(
            z_km,
            horizontal
            * (1.0 - eccentricity_squared * prime_vertical / (prime_vertical + altitude)),
        )
    return np.degrees(latitude), longitude, altitude


def fk_mapping_factor(
    elevation_deg: np.ndarray,
    orbit_radius_km: np.ndarray,
    shell_height_km: float,
) -> np.ndarray:
    """计算 F&K 厚壳映射因子 ``M=STEC/VTEC``。"""

    shell_radius_km = MEAN_EARTH_RADIUS_KM + shell_height_km
    zenith_rad = np.radians(90.0 - elevation_deg)
    ratio = shell_radius_km / orbit_radius_km
    radicand = ratio**2 - np.sin(zenith_rad) ** 2
    result = np.full_like(elevation_deg, np.nan, dtype=float)
    valid = radicand > 0.0
    result[valid] = (1.0 + ratio[valid]) / (
        np.sqrt(radicand[valid]) + np.cos(zenith_rad[valid])
    )
    return result


def _downsample_indices(time: np.ndarray, interval_seconds: float) -> np.ndarray:
    """按给定时间间隔选取观测索引，降低连续 POD 数据的采样密度。"""

    if interval_seconds <= 0.0 or time.size == 0:
        return np.arange(time.size)
    slots = np.floor((time - time[0]) / interval_seconds).astype(np.int64)
    _, first = np.unique(slots, return_index=True)
    return np.sort(first)


def read_pod_archive(
    archive_path: Path,
    *,
    elevation_min_deg: float,
    shell_height_km: float,
    sample_interval_seconds: float,
    constellations: set[str],
    max_arcs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取一个日 POD 压缩包，返回筛选后的观测点和逐文件审计表。

    该函数完成成员安全检查、数据质量筛选、坐标转换、映射因子计算与
    时间降采样；审计表记录每个成员被接受或拒绝的原因。
    """

    arrays: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "gps_seconds",
            "lat",
            "lon",
            "orbit_alt_km",
            "elevation_deg",
            "stec_tecu",
            "mapping_factor",
            "vtec_fk_tecu",
            "vtec_eq2_literal_tecu",
            "local_time_hour",
        )
    }
    stats: Counter[str] = Counter()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.name.startswith("podTc2_") or not member.name.endswith("_nc"):
                continue
            stats["arcs_seen"] += 1
            if max_arcs is not None and stats["arcs_seen"] > max_arcs:
                stats["arcs_seen"] -= 1
                break
            if not _safe_member(member):
                stats["arcs_unsafe"] += 1
                continue
            file_object = archive.extractfile(member)
            if file_object is None:
                stats["arcs_unreadable"] += 1
                continue
            try:
                payload = file_object.read()
                dataset = netCDF4.Dataset("podTc2_in_memory.nc", mode="r", memory=payload)
                try:
                    constellation = _attr_text(dataset, "conid", "G")[:1]
                    if constellation not in constellations:
                        stats["arcs_wrong_constellation"] += 1
                        continue
                    required_variables = {
                        "time", "TEC", "elevation", "x_LEO", "y_LEO", "z_LEO"
                    }
                    if not required_variables.issubset(dataset.variables):
                        stats["arcs_missing_variables"] += 1
                        continue
                    if (
                        _attr_int(dataset, "podflag") != 1
                        or _attr_int(dataset, "attflag") != 1
                        or _attr_int(dataset, "leodcb_flag") != 1
                    ):
                        stats["arcs_failed_quality_flags"] += 1
                        continue
                    # CDAAC 用 gpsdcb_flag 标记发射端 GPS DCB；下载的 GLONASS
                    # 弧段该标记未设置，因此默认仅保留 GPS 更为稳妥。
                    if constellation == "G" and _attr_int(dataset, "gpsdcb_flag") != 1:
                        stats["arcs_failed_quality_flags"] += 1
                        continue

                    time = _as_float(dataset.variables["time"])
                    stec = _as_float(dataset.variables["TEC"])
                    elevation = _as_float(dataset.variables["elevation"])
                    x_leo = _as_float(dataset.variables["x_LEO"])
                    y_leo = _as_float(dataset.variables["y_LEO"])
                    z_leo = _as_float(dataset.variables["z_LEO"])
                    lengths = {len(time), len(stec), len(elevation), len(x_leo), len(y_leo), len(z_leo)}
                    if len(lengths) != 1:
                        stats["arcs_length_mismatch"] += 1
                        continue
                    stats["raw_points"] += len(time)
                    valid = (
                        np.isfinite(time)
                        & np.isfinite(stec)
                        & np.isfinite(elevation)
                        & np.isfinite(x_leo)
                        & np.isfinite(y_leo)
                        & np.isfinite(z_leo)
                        & (stec >= 0.0)
                        & (stec <= 9999.0)
                        & (elevation > elevation_min_deg)
                    )
                    if not valid.any():
                        stats["arcs_without_high_elevation"] += 1
                        continue
                    time = time[valid]
                    stec = stec[valid]
                    elevation = elevation[valid]
                    x_leo = x_leo[valid]
                    y_leo = y_leo[valid]
                    z_leo = z_leo[valid]
                    stats["high_elevation_points"] += len(time)
                    keep = _downsample_indices(time, sample_interval_seconds)
                    time, stec, elevation = time[keep], stec[keep], elevation[keep]
                    x_leo, y_leo, z_leo = x_leo[keep], y_leo[keep], z_leo[keep]

                    orbit_radius = np.sqrt(x_leo**2 + y_leo**2 + z_leo**2)
                    latitude, longitude, altitude = ecef_to_geodetic(x_leo, y_leo, z_leo)
                    mapping = fk_mapping_factor(elevation, orbit_radius, shell_height_km)
                    valid_mapping = np.isfinite(mapping) & (mapping > 0.0)
                    if not valid_mapping.any():
                        stats["arcs_invalid_mapping"] += 1
                        continue
                    time = time[valid_mapping]
                    latitude = latitude[valid_mapping]
                    longitude = longitude[valid_mapping]
                    altitude = altitude[valid_mapping]
                    elevation = elevation[valid_mapping]
                    stec = stec[valid_mapping]
                    mapping = mapping[valid_mapping]
                    utc_seconds_of_day = np.mod(
                        time - GPS_MINUS_UTC_SECONDS_2021, 86400.0
                    )
                    local_time = np.mod(utc_seconds_of_day / 3600.0 + longitude / 15.0, 24.0)
                    values = {
                        "gps_seconds": time,
                        "lat": latitude,
                        "lon": longitude,
                        "orbit_alt_km": altitude,
                        "elevation_deg": elevation,
                        "stec_tecu": stec,
                        "mapping_factor": mapping,
                        "vtec_fk_tecu": stec / mapping,
                        "vtec_eq2_literal_tecu": stec * mapping,
                        "local_time_hour": local_time,
                    }
                    for name, value in values.items():
                        arrays[name].append(value)
                    stats["arcs_used"] += 1
                    stats["model_points"] += len(time)
                finally:
                    dataset.close()
            except (OSError, RuntimeError, TypeError, ValueError):
                stats["arcs_unreadable"] += 1
            finally:
                file_object.close()
    if not arrays["gps_seconds"]:
        raise ValueError("POD筛选后没有可用于建模的高仰角观测")
    points = pd.DataFrame(
        {name: np.concatenate(parts) for name, parts in arrays.items()}
    ).sort_values("gps_seconds", kind="stable", ignore_index=True)
    audit = pd.DataFrame(
        [{"item": key, "count": value} for key, value in sorted(stats.items())]
    )
    return points, audit


# 将离散 POD 观测按纬度和地方时分箱，建立可供掩星事件查询的二维模型。
def build_lat_local_time_model(
    points: pd.DataFrame, lat_bin_deg: float, local_time_bin_hour: float
) -> PodModel:
    data = points.copy()
    data["lat_bin"] = np.clip(
        np.floor((data["lat"] + 90.0) / lat_bin_deg).astype(int),
        0,
        int(round(180.0 / lat_bin_deg)) - 1,
    )
    data["local_time_bin"] = np.clip(
        np.floor(data["local_time_hour"] / local_time_bin_hour).astype(int),
        0,
        int(round(24.0 / local_time_bin_hour)) - 1,
    )
    grouped = data.groupby(["lat_bin", "local_time_bin"], sort=True)
    grid = grouped.agg(
        count=("vtec_fk_tecu", "size"),
        pod_vtec_fk_tecu=("vtec_fk_tecu", "median"),
        pod_vtec_eq2_literal_tecu=("vtec_eq2_literal_tecu", "median"),
        stec_median_tecu=("stec_tecu", "median"),
        elevation_median_deg=("elevation_deg", "median"),
    ).reset_index()
    grid["lat_center"] = -90.0 + (grid["lat_bin"] + 0.5) * lat_bin_deg
    grid["local_time_center"] = (
        grid["local_time_bin"] + 0.5
    ) * local_time_bin_hour
    features = _model_features(
        grid["lat_center"].to_numpy(), grid["local_time_center"].to_numpy()
    )
    return PodModel(grid=grid, tree=cKDTree(features), features=features)


# 用正余弦展开周期性的地方时，避免 0 时与 24 时在特征空间中被错误分开。
def _model_features(latitude: np.ndarray, local_time: np.ndarray) -> np.ndarray:
    angle = np.asarray(local_time, dtype=float) * (2.0 * np.pi / 24.0)
    cyclic_radius = 180.0 / np.pi
    return np.column_stack(
        (
            np.asarray(latitude, dtype=float),
            cyclic_radius * np.cos(angle),
            cyclic_radius * np.sin(angle),
        )
    )


# 利用邻近网格对目标事件插值，同时返回距离用于判断结果是否可靠。
def interpolate_model(
    model: PodModel,
    latitude: np.ndarray,
    local_time: np.ndarray,
    *,
    neighbors: int,
    max_distance_deg: float,
) -> pd.DataFrame:
    query = _model_features(latitude, local_time)
    k = min(neighbors, len(model.grid))
    distance, index = model.tree.query(query, k=k)
    if k == 1:
        distance, index = distance[:, None], index[:, None]
    result: dict[str, np.ndarray] = {}
    for source, destination in (
        ("pod_vtec_fk_tecu", "pod_vtec_fk"),
        ("pod_vtec_eq2_literal_tecu", "pod_vtec_eq2_literal"),
    ):
        grid_values = model.grid[source].to_numpy(dtype=float)
        values = np.full(len(query), np.nan)
        for row in range(len(query)):
            use = np.isfinite(distance[row]) & (distance[row] <= max_distance_deg)
            if not use.any():
                continue
            selected_distance = distance[row, use]
            selected_values = grid_values[index[row, use]]
            weights = 1.0 / np.maximum(selected_distance, 0.25) ** 2
            values[row] = float(np.average(selected_values, weights=weights))
        result[destination] = values
    result["pod_interp_nearest_distance_deg"] = distance[:, 0]
    result["pod_interp_neighbor_count"] = np.sum(distance <= max_distance_deg, axis=1)
    return pd.DataFrame(result)


# 读取掩星 CSV，将两种 POD 顶部 VTEC 结果和诊断字段写入新的数据表。
def attach_to_ro_csv(
    csv_path: Path,
    output_path: Path,
    model: PodModel,
    *,
    neighbors: int,
    max_distance_deg: float,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"time", "lat", "lon", "tec0"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path.name} 缺少字段：{', '.join(missing)}")
    time = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    numeric = frame[["lat", "lon", "tec0"]].apply(pd.to_numeric, errors="coerce")
    valid = time.notna() & np.isfinite(numeric.to_numpy()).all(axis=1)
    local_time = np.mod(
        time.dt.hour
        + time.dt.minute / 60.0
        + time.dt.second / 3600.0
        + numeric["lon"] / 15.0,
        24.0,
    )
    interpolation = pd.DataFrame(
        {
            "pod_vtec_fk": np.nan,
            "pod_vtec_eq2_literal": np.nan,
            "pod_interp_nearest_distance_deg": np.nan,
            "pod_interp_neighbor_count": 0,
        },
        index=frame.index,
    )
    interpolation.loc[valid, :] = interpolate_model(
        model,
        numeric.loc[valid, "lat"].to_numpy(),
        local_time.loc[valid].to_numpy(),
        neighbors=neighbors,
        max_distance_deg=max_distance_deg,
    ).to_numpy()
    frame["local_time_hour"] = local_time
    for column in interpolation.columns:
        frame[column] = interpolation[column]
    frame["pod_vtec"] = frame["pod_vtec_fk"]
    frame["tec0_plus_pod_vtec"] = numeric["tec0"] + frame["pod_vtec"]
    frame["tec0_plus_pod_eq2_literal"] = (
        numeric["tec0"] + frame["pod_vtec_eq2_literal"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.10g")
    os.replace(temporary, output_path)
    return frame


# 绘制 POD 筛选、网格覆盖及插值效果，便于检查每日处理质量。
def plot_diagnostics(
    points: pd.DataFrame,
    model: PodModel,
    events: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    sample = points.sample(min(50_000, len(points)), random_state=42)
    scatter = axes[0, 0].scatter(
        sample["local_time_hour"], sample["lat"], c=sample["vtec_fk_tecu"],
        s=4, cmap="turbo", vmin=0.0,
        vmax=float(np.nanpercentile(points["vtec_fk_tecu"], 98.0)), rasterized=True,
    )
    axes[0, 0].set(xlabel="Local time (hour)", ylabel="LEO latitude (deg)", title="High-elevation POD observations")
    fig.colorbar(scatter, ax=axes[0, 0], label="F&K VTEC (TECU)")

    grid = model.grid
    scatter_grid = axes[0, 1].scatter(
        grid["local_time_center"], grid["lat_center"],
        c=grid["pod_vtec_fk_tecu"], s=np.clip(grid["count"], 10, 150),
        cmap="turbo", vmin=0.0,
        vmax=float(np.nanpercentile(grid["pod_vtec_fk_tecu"], 98.0)),
    )
    axes[0, 1].set(xlabel="Local time (hour)", ylabel="Latitude bin (deg)", title="Median latitude/local-time model")
    fig.colorbar(scatter_grid, ax=axes[0, 1], label="F&K VTEC (TECU)")

    axes[1, 0].hist(points["vtec_fk_tecu"], bins=80, alpha=0.65, label="Standard: STEC / M")
    axes[1, 0].hist(points["vtec_eq2_literal_tecu"], bins=80, alpha=0.55, label="Paper Eq. (2): STEC × M")
    axes[1, 0].set(xlabel="POD VTEC (TECU)", ylabel="Samples", title="Mapping convention sensitivity")
    axes[1, 0].legend()

    if "tec1" in events.columns:
        valid = (
            np.isfinite(
                events[["tec1", "pod_vtec_fk", "pod_vtec_eq2_literal"]].to_numpy()
            ).all(axis=1)
            & ~np.isclose(events["tec1"], -999.0, rtol=0.0, atol=1.0e-9)
        )
        valid_tec1 = events.loc[valid, "tec1"]
        if len(valid_tec1) >= 100:
            low, high = valid_tec1.quantile([0.01, 0.99])
            valid &= events["tec1"].between(low, high)
        axes[1, 1].scatter(events.loc[valid, "tec1"], events.loc[valid, "pod_vtec_fk"], s=8, alpha=0.45, label="Standard F&K")
        axes[1, 1].scatter(events.loc[valid, "tec1"], events.loc[valid, "pod_vtec_eq2_literal"], s=8, alpha=0.35, label="Paper Eq. (2)")
        axes[1, 1].set(
            xlabel="CDAAC ionPrf tec1 (TECU)",
            ylabel="Interpolated POD VTEC (TECU)",
            title="Independent topside estimates (central 98% valid tec1)",
        )
        axes[1, 1].legend()
    else:
        axes[1, 1].set_axis_off()
    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def comparison_metrics(events: pd.DataFrame, outlier_fraction: float = 0.001) -> pd.DataFrame:
    """比较不同顶部 TEC 补偿与 GIM，并明确采用 ``GIM-估计值`` 的偏差符号。"""

    rows: list[dict[str, object]] = []
    for column, label in (
        ("tec1", "CDAAC ionPrf tec1"),
        ("pod_vtec_fk", "POD standard F&K: STEC/M"),
        ("pod_vtec_eq2_literal", "POD Wu et al. Eq. (2) literal: STEC*M"),
    ):
        valid = np.isfinite(events[["tec0", column, "gim_vtec"]].to_numpy()).all(axis=1)
        if column == "tec1":
            valid &= ~np.isclose(events[column], -999.0, rtol=0.0, atol=1.0e-9)
        selected = events.loc[valid, ["tec0", column, "gim_vtec"]].copy()
        total = selected["tec0"] + selected[column]
        residual = selected["gim_vtec"] - total
        for method, keep in (
            ("raw", np.ones(len(selected), dtype=bool)),
            (
                "remove largest 0.1% absolute residual",
                residual.abs() <= residual.abs().quantile(1.0 - outlier_fraction),
            ),
        ):
            x = total.loc[keep].to_numpy(dtype=float)
            y = selected.loc[keep, "gim_vtec"].to_numpy(dtype=float)
            difference = y - x
            rows.append(
                {
                    "topside_column": column,
                    "method": label,
                    "filter": method,
                    "count": len(x),
                    "mean_topside_tecu": float(selected.loc[keep, column].mean()),
                    "r_total_vs_gim": float(np.corrcoef(x, y)[0, 1]),
                    "bias_gim_minus_total_tecu": float(np.mean(difference)),
                    "rmse_total_vs_gim_tecu": float(np.sqrt(np.mean(difference**2))),
                }
            )
    return pd.DataFrame(rows)


def _archive_year_doy(path: Path) -> tuple[int, int]:
    """从 POD 压缩包文件名解析年份和年积日。"""

    match = POD_ARCHIVE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"无法从POD文件名识别年份/DOY：{path.name}")
    return int(match.group(1)), int(match.group(2))


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""

    parser = argparse.ArgumentParser(description="POD STEC转轨道上方VTEC并写入COSMIC事件CSV")
    parser.add_argument("--pod-archive", type=Path, default=None)
    parser.add_argument("--ro-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--diagnostic-dir", type=Path, default=None)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--doy", type=int, default=75)
    parser.add_argument("--elevation-min", type=float, default=50.0)
    parser.add_argument("--shell-height-km", type=float, default=2100.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=30.0)
    parser.add_argument("--constellations", default="G", help="默认仅使用绝对GPS TEC；如需GLONASS可填GR")
    parser.add_argument("--lat-bin-deg", type=float, default=5.0)
    parser.add_argument("--local-time-bin-hour", type=float, default=1.0)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-distance-deg", type=float, default=15.0)
    parser.add_argument("--max-arcs", type=int, default=None, help="仅用于快速测试")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """串联 POD 读取、建模、掩星插值、指标计算和结果输出。"""

    args = parse_args()
    if not 0.0 <= args.elevation_min < 90.0:
        raise SystemExit("elevation-min必须位于[0, 90)")
    if args.shell_height_km <= 0.0:
        raise SystemExit("shell-height-km必须为正数")
    if args.lat_bin_deg <= 0.0 or 180.0 % args.lat_bin_deg != 0.0:
        raise SystemExit("lat-bin-deg必须能整除180")
    if args.local_time_bin_hour <= 0.0 or 24.0 % args.local_time_bin_hour != 0.0:
        raise SystemExit("local-time-bin-hour必须能整除24")
    if args.neighbors < 1 or args.max_distance_deg <= 0.0:
        raise SystemExit("neighbors和max-distance-deg必须为正数")
    constellations = {value for value in args.constellations.upper() if value in {"G", "R"}}
    if not constellations:
        raise SystemExit("constellations至少包含G或R")

    pod_archive = args.pod_archive or DEFAULT_POD_DIR / f"podTc2_nrt_{args.year}_{args.doy:03d}.tar.gz"
    if not pod_archive.is_file():
        raise SystemExit(f"POD压缩包不存在：{pod_archive}")
    year, doy = _archive_year_doy(pod_archive)
    ro_csv = args.ro_csv or DEFAULT_CSV_DIR / f"cosmic2_ionPrF_{year}_{doy:03d}.csv"
    if not ro_csv.is_file():
        raise SystemExit(f"掩星CSV不存在：{ro_csv}")
    output_csv = args.output_csv or DEFAULT_OUTPUT_DIR / f"cosmic2_ionPrF_{year}_{doy:03d}_pod.csv"
    diagnostic_dir = args.diagnostic_dir or DEFAULT_DIAGNOSTIC_DIR / f"DOY{doy:03d}"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取POD：{pod_archive}")
    points, audit = read_pod_archive(
        pod_archive,
        elevation_min_deg=args.elevation_min,
        shell_height_km=args.shell_height_km,
        sample_interval_seconds=args.sample_interval_seconds,
        constellations=constellations,
        max_arcs=args.max_arcs,
    )
    print(audit.to_string(index=False))
    model = build_lat_local_time_model(points, args.lat_bin_deg, args.local_time_bin_hour)
    events = attach_to_ro_csv(
        ro_csv, output_csv, model,
        neighbors=args.neighbors, max_distance_deg=args.max_distance_deg,
    )
    points.to_csv(diagnostic_dir / "pod_points_downsampled.csv", index=False, encoding="utf-8-sig")
    model.grid.to_csv(diagnostic_dir / "pod_lat_localtime_model.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(diagnostic_dir / "pod_processing_audit.csv", index=False, encoding="utf-8-sig")
    metrics = comparison_metrics(events)
    metrics.to_csv(
        diagnostic_dir / "pod_method_comparison.csv", index=False, encoding="utf-8-sig"
    )
    plot_diagnostics(points, model, events, diagnostic_dir / "pod_vtec_diagnostics.png", args.dpi)
    summary = pd.DataFrame(
        [
            {"item": "ro_rows", "value": len(events)},
            {"item": "ro_rows_with_pod_vtec", "value": int(events["pod_vtec"].notna().sum())},
            {"item": "pod_model_points", "value": len(points)},
            {"item": "pod_model_cells", "value": len(model.grid)},
            {"item": "elevation_min_deg", "value": args.elevation_min},
            {"item": "shell_height_km", "value": args.shell_height_km},
            {"item": "mapping_recommended", "value": "VTEC = STEC / M"},
            {"item": "paper_equation_2_literal", "value": "VTEC = STEC * M"},
        ]
    )
    summary.to_csv(diagnostic_dir / "run_summary.csv", index=False, encoding="utf-8-sig")
    print(f"输出CSV：{output_csv.resolve()}")
    print(f"诊断目录：{diagnostic_dir.resolve()}")


if __name__ == "__main__":
    main()
