"""生成可直接用于建模的 COSMIC-2 2021 年逐日特征 CSV。

每条 ionPrf 剖面先执行 ``Cosmic_process.py`` 中已有的四项质量控制，再按
``cosmic2021_ionprf_gim.py`` 的规则剔除降轨机动期和轨道高度异常记录，并
检查电子密度积分是否有效。通过筛选的记录会补充准偶极地磁坐标、周期特征、
GIM VTEC、OMNI Kp/Dst/F10.7 和陆海标记。陆地与海洋数据均予以保留。

GIM 使用 IONEX 文件自身的经度轴决定归一化范围，并在包含 -180/180 重复端点
时统一映射到左端点，避免 0/360 或 ±180 度约定不一致造成空间插值错误。
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from apexpy import Apex
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from helper.interpolator import GIMInterpolator  # noqa: E402

import Cosmic_process as existing_qc  # noqa: E402
import analyze_cosmic_tec_gim as land_tools  # noqa: E402
import cosmic2021_ionprf_gim as base  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "modeling_csv_qc"
DEFAULT_GIM_DIR = PROJECT_ROOT / "Data" / "GIM2021" / "IGS"
DEFAULT_OMNI_PATH = PROJECT_ROOT / "Data" / "Other" / "omni2_2021.dat.csv"
DEFAULT_LANDMASK_PATH = PROJECT_ROOT / "Data" / "Other" / "landmask_static.nc"

MAX_INDEX_GAP = {
    "f10.7_Index": pd.Timedelta(hours=72),
    "Dst_Index": pd.Timedelta(hours=6),
    "Kp_Index": pd.Timedelta(hours=30),
}

OUTPUT_COLUMNS = [
    "datetime",
    "lat",
    "lon",
    "mag_lat",
    "mag_lon",
    "lon_s",
    "lon_c",
    "mag_lat_s",
    "mag_lat_c",
    "mag_lon_s",
    "mag_lon_c",
    "DOY_s",
    "DOY_c",
    "local_time_s",
    "local_time_c",
    "tec0",
    "gim_vtec",
    "gim_tec",
    "Kp_Index",
    "Dst_Index",
    "f10.7_Index",
    "land_fraction",
    "is_land",
    "surface_type",
]

AUDIT_COLUMNS = [
    "source_archive",
    "source_member",
    "datetime",
    "existing_qc_pass",
    "orbit_stage_pass",
    "integration_pass",
    "final_status",
    "reject_reason",
    "detail",
]


@dataclass(frozen=True)
class OmniData:
    """保存已清洗的 OMNI 时间轴和三个空间天气指数。"""

    timestamps: np.ndarray
    values: dict[str, np.ndarray]


def _as_float(variable: netCDF4.Variable) -> np.ndarray:
    """把 NetCDF 变量读为一维浮点数组，并将掩码值转为 NaN。"""

    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def read_profile(payload: bytes) -> dict[str, object]:
    """一次性读取元数据、剖面及既有质控所需字段。"""

    try:
        dataset = netCDF4.Dataset("ionPrf_in_memory.nc", mode="r", memory=payload)
    except (OSError, RuntimeError) as exc:
        raise base.ProfileRejected("unreadable_netcdf", str(exc)) from exc

    try:
        required_attrs = (
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "edmaxtime",
            "edmaxlat",
            "edmaxlon",
            "edmaxalt",
            "edmax",
            "tec0",
            "tec1",
        )
        missing_attrs = [name for name in required_attrs if name not in dataset.ncattrs()]
        missing_vars = [
            name
            for name in ("MSL_alt", "ELEC_dens")
            if name not in dataset.variables
        ]
        if missing_attrs or missing_vars:
            raise base.ProfileRejected(
                "missing_required_field", ",".join(missing_attrs + missing_vars)
            )

        gps_seconds = base._finite_float(dataset.getncattr("edmaxtime"), "edmaxtime")
        time_utc = base.gps_seconds_to_utc(gps_seconds)
        gps_calendar = base.GPS_EPOCH + pd.to_timedelta(gps_seconds, unit="s")
        calendar = base._calendar_time(dataset)
        if abs((calendar - gps_calendar).total_seconds()) > 1.0:
            raise base.ProfileRejected(
                "invalid_time_or_position", "edmaxtime/calendar mismatch"
            )

        lat = base._finite_float(dataset.getncattr("edmaxlat"), "edmaxlat")
        lon = base._finite_float(dataset.getncattr("edmaxlon"), "edmaxlon")
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise base.ProfileRejected(
                "invalid_time_or_position", "lat/lon out of range"
            )

        altitude = _as_float(dataset.variables["MSL_alt"])
        density = _as_float(dataset.variables["ELEC_dens"])
        if altitude.size != density.size:
            raise base.ProfileRejected(
                "missing_required_field", "profile length mismatch"
            )

        hm_f2 = base._finite_float(dataset.getncattr("edmaxalt"), "edmaxalt")
        nm_f2 = base._finite_float(dataset.getncattr("edmax"), "edmax")
        attrs = {
            name: dataset.getncattr(name)
            for name in ("edorbalt", "topalt")
            if name in dataset.ncattrs()
        }
        return {
            "time_utc": time_utc,
            "lat": lat,
            "lon": lon,
            "hm_f2": hm_f2,
            "tec0": base._finite_float(dataset.getncattr("tec0"), "tec0"),
            "altitude": altitude,
            "density": density,
            "attrs": attrs,
            "qc_profile": {
                "alt": altitude,
                "ne": density,
                "ne_smooth": uniform_filter1d(density, size=9, mode="nearest"),
                "nm_f2_raw": nm_f2,
                "hm_f2_raw": hm_f2,
            },
        }
    finally:
        dataset.close()


def load_omni(path: Path) -> OmniData:
    """读取 OMNI CSV，处理填充值并生成有序 UTC 时间轴。"""

    frame = pd.read_csv(path)
    required = {
        "Year",
        "Decimal_Day",
        "Hour",
        "f10.7_Index",
        "Dst_Index",
        "Kp_Index",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OMNI 文件缺少字段：{', '.join(missing)}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Year", "Decimal_Day", "Hour"]).copy()
    base_dates = pd.to_datetime(
        frame["Year"].astype(int).astype(str) + "-01-01", errors="coerce"
    )
    frame["datetime"] = (
        base_dates
        + pd.to_timedelta(frame["Decimal_Day"] - 1.0, unit="D")
        + pd.to_timedelta(frame["Hour"], unit="h")
    )
    frame["f10.7_Index"] = frame["f10.7_Index"].replace(999.9, np.nan)
    frame["Dst_Index"] = frame["Dst_Index"].replace(99999, np.nan)
    frame["Kp_Index"] = frame["Kp_Index"].replace(99, np.nan)
    kp_valid = frame["Kp_Index"].dropna()
    if not kp_valid.empty and kp_valid.quantile(0.95) > 9.0:
        frame["Kp_Index"] = frame["Kp_Index"] / 10.0

    limits = {
        "f10.7_Index": (0.0, 500.0),
        "Dst_Index": (-2500.0, 1000.0),
        "Kp_Index": (0.0, 9.0),
    }
    for column, (lower, upper) in limits.items():
        invalid = frame[column].notna() & ~frame[column].between(lower, upper)
        frame.loc[invalid, column] = np.nan

    value_columns = ["f10.7_Index", "Dst_Index", "Kp_Index"]
    frame = (
        frame[["datetime", *value_columns]]
        .dropna(subset=["datetime"])
        .groupby("datetime", as_index=False, sort=True)
        .median(numeric_only=True)
    )
    timestamps = frame["datetime"].astype("int64").to_numpy(dtype=np.float64) / 1e9
    return OmniData(
        timestamps=timestamps,
        values={name: frame[name].to_numpy(dtype=float) for name in value_columns},
    )


def safe_time_interp(
    targets: np.ndarray,
    source_time: np.ndarray,
    source_value: np.ndarray,
    max_gap: pd.Timedelta,
) -> np.ndarray:
    """仅在有效观测包围且时间缺口允许时进行一维线性插值。"""

    targets = np.asarray(targets, dtype=np.float64)
    valid = np.isfinite(source_time) & np.isfinite(source_value)
    x = source_time[valid]
    y = source_value[valid]
    result = np.full(targets.shape, np.nan, dtype=float)
    if x.size == 0:
        return result

    order = np.argsort(x)
    x, y = x[order], y[order]
    positions = np.searchsorted(x, targets, side="left")
    clipped = np.minimum(positions, x.size - 1)
    exact = (positions < x.size) & (x[clipped] == targets)
    result[exact] = y[clipped[exact]]

    between = (~exact) & (positions > 0) & (positions < x.size)
    selected = np.flatnonzero(between)
    if selected.size == 0:
        return result
    right = positions[selected]
    left = right - 1
    gaps = x[right] - x[left]
    allowed = gaps <= max_gap.total_seconds()
    selected, left, right = selected[allowed], left[allowed], right[allowed]
    if selected.size:
        weights = (targets[selected] - x[left]) / (x[right] - x[left])
        result[selected] = y[left] + weights * (y[right] - y[left])
    return result


def normalize_longitudes(longitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
    """按 GIM 经度轴约定归一化经度，并统一周期端点。"""

    longitude = np.asarray(longitude, dtype=float)
    grid_longitude = np.asarray(grid_longitude, dtype=float)
    lower = float(np.min(grid_longitude))
    upper = float(np.max(grid_longitude))
    span = upper - lower
    if span < 359.0 or span > 361.0:
        raise ValueError(f"GIM 经度轴未覆盖完整全球：[{lower}, {upper}]")
    normalized = np.mod(longitude - lower, 360.0) + lower
    normalized[np.isclose(normalized, upper, atol=1e-10)] = lower
    return normalized


def interpolate_gim(
    gim: GIMInterpolator,
    times: pd.Series,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """按实际 UTC 日期分组，对 GIM 做经度安全的三维时空线性插值。"""

    timestamps = pd.to_datetime(times, utc=True, errors="coerce").dt.tz_localize(None)
    result = np.full(len(timestamps), np.nan, dtype=float)
    dates = timestamps.dt.normalize()
    for date_value in dates.dropna().unique():
        selected = (dates == date_value).to_numpy()
        target_time = timestamps.loc[selected]
        try:
            data = gim.load_day(pd.Timestamp(date_value).to_pydatetime())
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            continue

        grid_times = pd.to_datetime(data["times"])
        time_axis = (
            grid_times.astype("int64").to_numpy(dtype=np.float64) / 1e9
        )
        grid_latitude = np.asarray(data["lats"], dtype=float)
        grid_longitude = np.asarray(data["lons"], dtype=float)
        grid = np.asarray(data["tec"], dtype=float)

        if grid_latitude[0] > grid_latitude[-1]:
            grid_latitude = grid_latitude[::-1]
            grid = grid[:, ::-1, :]
        if grid_longitude[0] > grid_longitude[-1]:
            grid_longitude = grid_longitude[::-1]
            grid = grid[:, :, ::-1]

        target_lon = normalize_longitudes(longitude[selected], grid_longitude)
        target_seconds = (
            target_time.astype("int64").to_numpy(dtype=np.float64) / 1e9
        )
        query = np.column_stack(
            (target_seconds, latitude[selected], target_lon)
        )
        interpolator = RegularGridInterpolator(
            (time_axis, grid_latitude, grid_longitude),
            grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        result[selected] = interpolator(query)
    return result


def add_model_features(
    frame: pd.DataFrame,
    omni: OmniData,
    landmask: land_tools.LandMask,
    gim: GIMInterpolator,
) -> pd.DataFrame:
    """为通过质控的记录添加建模所需的全部派生特征。"""

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)

    # 地磁场年内变化缓慢；与 Jason 流程一致，每个日文件使用一个参考时刻。
    apex = Apex(date=data["datetime"].iloc[0].to_pydatetime())
    mag_lat, mag_lon = apex.geo2qd(
        data["lat"].to_numpy(dtype=float),
        data["lon"].to_numpy(dtype=float),
        height=data["hm_f2"].to_numpy(dtype=float),
    )
    data["mag_lat"] = mag_lat
    data["mag_lon"] = (np.asarray(mag_lon, dtype=float) + 180.0) % 360.0 - 180.0

    lon_angle = np.deg2rad(np.mod(data["lon"].to_numpy(dtype=float), 360.0))
    mag_lon_angle = np.deg2rad(
        np.mod(data["mag_lon"].to_numpy(dtype=float), 360.0)
    )
    mag_lat_angle = np.deg2rad(data["mag_lat"].to_numpy(dtype=float))
    data["lon_s"], data["lon_c"] = np.sin(lon_angle), np.cos(lon_angle)
    data["mag_lon_s"], data["mag_lon_c"] = (
        np.sin(mag_lon_angle),
        np.cos(mag_lon_angle),
    )
    data["mag_lat_s"], data["mag_lat_c"] = (
        np.sin(mag_lat_angle),
        np.cos(mag_lat_angle),
    )

    doy = data["datetime"].dt.dayofyear.to_numpy(dtype=float)
    doy_phase = 2.0 * np.pi * doy / 365.25
    data["DOY_s"], data["DOY_c"] = np.sin(doy_phase), np.cos(doy_phase)
    utc_hour = (
        data["datetime"].dt.hour.to_numpy(dtype=float)
        + data["datetime"].dt.minute.to_numpy(dtype=float) / 60.0
        + data["datetime"].dt.second.to_numpy(dtype=float) / 3600.0
        + data["datetime"].dt.microsecond.to_numpy(dtype=float) / 3.6e9
    )
    local_hour = np.mod(utc_hour + data["lon"].to_numpy(dtype=float) / 15.0, 24.0)
    local_phase = 2.0 * np.pi * local_hour / 24.0
    data["local_time_s"] = np.sin(local_phase)
    data["local_time_c"] = np.cos(local_phase)

    data["gim_vtec"] = interpolate_gim(
        gim,
        data["datetime"],
        data["lat"].to_numpy(dtype=float),
        data["lon"].to_numpy(dtype=float),
    )
    data["gim_tec"] = data["gim_vtec"]

    target_seconds = (
        data["datetime"].astype("int64").to_numpy(dtype=np.float64) / 1e9
    )
    for column in ("Kp_Index", "Dst_Index", "f10.7_Index"):
        data[column] = safe_time_interp(
            target_seconds,
            omni.timestamps,
            omni.values[column],
            MAX_INDEX_GAP[column],
        )

    data["land_fraction"] = landmask.fraction_at(
        data["lat"].to_numpy(dtype=float), data["lon"].to_numpy(dtype=float)
    )
    data["is_land"] = (data["land_fraction"] >= landmask.threshold).astype(np.int8)
    data["surface_type"] = np.where(data["is_land"] == 1, "land", "ocean")
    data["datetime"] = data["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return data[OUTPUT_COLUMNS]


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    """原子写入 CSV，防止中断时留下不完整的正式结果。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.10g")
    os.replace(temporary, destination)


def process_archive(
    archive_path: Path,
    output_dir: Path,
    omni: OmniData,
    landmask: land_tools.LandMask,
    gim: GIMInterpolator,
    *,
    orbit_alt_min: float,
    orbit_alt_max: float,
    max_profiles: int | None = None,
) -> Counter:
    """质控一个年积日压缩包并输出建模 CSV 与逐剖面审计表。"""

    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    record_audit_indices: list[int] = []
    stats: Counter = Counter()
    seen: set[str] = set()

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            name = member.name.replace("\\", "/")
            if not name.startswith("ionPrf_") or not name.endswith("_nc"):
                continue
            stats["total"] += 1
            audit = {
                "source_archive": archive_path.name,
                "source_member": name,
                "datetime": "",
                "existing_qc_pass": False,
                "orbit_stage_pass": False,
                "integration_pass": False,
                "final_status": "rejected",
                "reject_reason": "",
                "detail": "",
            }
            try:
                if not base._member_is_safe(member):
                    raise base.ProfileRejected("unsafe_archive_member")
                if name in seen:
                    raise base.ProfileRejected("duplicate_event", "duplicate member name")
                seen.add(name)
                file_object = archive.extractfile(member)
                if file_object is None:
                    raise base.ProfileRejected("unreadable_netcdf", "extractfile returned None")
                with file_object:
                    profile = read_profile(file_object.read())

                time_utc = profile["time_utc"]
                assert isinstance(time_utc, datetime)
                audit["datetime"] = base._iso_utc(time_utc)
                leo_id = base.parse_leo_id(name)
                passed, reason = existing_qc.quality_control(
                    profile["qc_profile"], existing_qc.CosmicConfig()
                )
                audit["existing_qc_pass"] = passed
                if not passed:
                    raise base.ProfileRejected("failed_existing_qc", reason)
                stats["existing_qc_pass"] += 1

                if base.in_maneuver_window(leo_id, time_utc):
                    raise base.ProfileRejected("official_maneuver_window")
                altitude = profile["altitude"]
                density = profile["density"]
                assert isinstance(altitude, np.ndarray)
                assert isinstance(density, np.ndarray)
                orbit_alt, _ = base.choose_orbit_altitude(profile["attrs"], altitude)
                if not orbit_alt_min <= orbit_alt <= orbit_alt_max:
                    raise base.ProfileRejected(
                        "orbit_alt_outside_configured_range",
                        f"{orbit_alt:.6g} km not in [{orbit_alt_min:g}, {orbit_alt_max:g}]",
                    )
                audit["orbit_stage_pass"] = True
                stats["orbit_stage_pass"] += 1

                # 与既有流程一致：积分值虽不作为本表特征，仍用于检查剖面是否可积。
                base.integrate_electron_density(altitude, density)
                audit["integration_pass"] = True
                stats["integration_pass"] += 1
                records.append(
                    {
                        "datetime": audit["datetime"],
                        "lat": profile["lat"],
                        "lon": profile["lon"],
                        "hm_f2": profile["hm_f2"],
                        "tec0": profile["tec0"],
                    }
                )
                # 此时只表示剖面级检查通过；派生特征完整后才标记最终通过。
                audit["final_status"] = "profile_passed"
                record_audit_indices.append(len(audits))
            except base.ProfileRejected as exc:
                audit["reject_reason"] = exc.reason
                audit["detail"] = exc.detail.replace("\r", " ").replace("\n", " ")[:500]
                stats[exc.reason] += 1
            except Exception as exc:
                audit["reject_reason"] = "unexpected_profile_error"
                audit["detail"] = str(exc).replace("\r", " ").replace("\n", " ")[:500]
                stats["unexpected_profile_error"] += 1
            audits.append(audit)

            if stats["total"] % 500 == 0:
                print(
                    f"    已读 {stats['total']}，原质控通过 {stats['existing_qc_pass']}，"
                    f"轨道通过 {stats['orbit_stage_pass']}，剖面可积 {stats['integration_pass']}"
                )
            if max_profiles is not None and stats["total"] >= max_profiles:
                break

    raw = pd.DataFrame.from_records(records)
    model = add_model_features(raw, omni, landmask, gim)
    finite_required = np.isfinite(
        model[
            [
                "lat",
                "lon",
                "mag_lat",
                "mag_lon",
                "tec0",
                "gim_vtec",
                "Kp_Index",
                "Dst_Index",
                "f10.7_Index",
                "land_fraction",
            ]
        ].to_numpy(dtype=float)
    ).all(axis=1) if not model.empty else np.array([], dtype=bool)
    stats["feature_rejected"] = int((~finite_required).sum())
    for record_index, valid in enumerate(finite_required):
        audit = audits[record_audit_indices[record_index]]
        if valid:
            audit["final_status"] = "passed"
        else:
            audit["final_status"] = "rejected"
            audit["reject_reason"] = "feature_interpolation_failed"
            audit["detail"] = "GIM/OMNI/地磁坐标或陆海特征存在非有限值"
    if not model.empty:
        model = model.loc[finite_required].sort_values("datetime", kind="stable")
        model = model.reset_index(drop=True)
    stats["output"] = len(model)

    match = base.re.search(r"_(\d{4})_(\d{3})\.tar\.gz$", archive_path.name)
    stem = (
        f"cosmic2_model_{match.group(1)}_{match.group(2)}"
        if match
        else archive_path.name.removesuffix(".tar.gz")
    )
    _atomic_csv(model, output_dir / f"{stem}.csv")
    _atomic_csv(
        pd.DataFrame.from_records(audits, columns=AUDIT_COLUMNS),
        output_dir / f"{stem}_qc_log.csv",
    )
    return stats


def validate_gim_longitude(gim: GIMInterpolator) -> None:
    """检查 GIM 周期接缝以及向量化插值与既有单点实现的一致性。"""

    test_time = datetime(2021, 1, 1, 12)
    data = gim.load_day(test_time)
    grid = np.asarray(data["tec"], dtype=float)
    lons = np.asarray(data["lons"], dtype=float)
    if np.isclose(lons[-1] - lons[0], 360.0):
        seam_error = float(np.nanmax(np.abs(grid[:, :, 0] - grid[:, :, -1])))
        if seam_error > 1e-6:
            raise ValueError(f"GIM ±180° 接缝不连续，最大差值 {seam_error:g} TECU")

    sample = pd.Series(pd.to_datetime(["2021-01-01T12:00:00Z"]), name="datetime")
    vector_value = interpolate_gim(
        gim, sample, np.array([0.0]), np.array([180.1])
    )[0]
    scalar_value = gim.query(test_time, 0.0, -179.9)
    if not np.isclose(vector_value, scalar_value, atol=1e-10, rtol=1e-10):
        raise ValueError(
            f"GIM 经度归一化验证失败：{vector_value} != {scalar_value}"
        )


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 COSMIC-2 逐日建模特征 CSV")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gim-dir", type=Path, default=DEFAULT_GIM_DIR)
    parser.add_argument("--omni", type=Path, default=DEFAULT_OMNI_PATH)
    parser.add_argument("--landmask", type=Path, default=DEFAULT_LANDMASK_PATH)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--start-doy", type=int, default=1)
    parser.add_argument("--end-doy", type=int, default=365)
    parser.add_argument("--land-threshold", type=float, default=0.5)
    parser.add_argument("--orbit-alt-min", type=float, default=base.DEFAULT_ORBIT_ALT_MIN_KM)
    parser.add_argument("--orbit-alt-max", type=float, default=base.DEFAULT_ORBIT_ALT_MAX_KM)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="覆盖已存在的逐日结果")
    return parser.parse_args()


def main() -> None:
    """验证参考数据后，逐年积日生成建模数据和质控审计文件。"""

    args = parse_args()
    if not 1 <= args.start_doy <= args.end_doy <= 365:
        raise SystemExit("DOY 范围必须满足 1 <= start-doy <= end-doy <= 365")
    for path, label in (
        (args.input_dir, "COSMIC 目录"),
        (args.gim_dir, "GIM 目录"),
        (args.omni, "OMNI 文件"),
        (args.landmask, "陆海掩膜"),
    ):
        if not path.exists():
            raise SystemExit(f"{label}不存在：{path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    omni = load_omni(args.omni)
    landmask = land_tools.load_landmask(args.landmask, args.land_threshold)
    gim = GIMInterpolator(args.gim_dir)
    validate_gim_longitude(gim)
    print("GIM 经度范围与 ±180° 周期接缝验证通过")

    total: Counter = Counter()
    failures: list[tuple[int, str]] = []
    for position, doy in enumerate(range(args.start_doy, args.end_doy + 1), start=1):
        archive = args.input_dir / f"ionPrf_prov1_{args.year}_{doy:03d}.tar.gz"
        output = args.output_dir / f"cosmic2_model_{args.year}_{doy:03d}.csv"
        if output.is_file() and not args.force:
            print(f"[{position}/{args.end_doy - args.start_doy + 1}] DOY {doy:03d} 已存在，跳过")
            continue
        if not archive.is_file():
            failures.append((doy, "输入压缩包不存在"))
            print(f"[{position}/{args.end_doy - args.start_doy + 1}] DOY {doy:03d} 缺失")
            continue
        print(f"[{position}/{args.end_doy - args.start_doy + 1}] DOY {doy:03d} 开始")
        try:
            stats = process_archive(
                archive,
                args.output_dir,
                omni,
                landmask,
                gim,
                orbit_alt_min=args.orbit_alt_min,
                orbit_alt_max=args.orbit_alt_max,
                max_profiles=args.max_profiles,
            )
            total.update(stats)
            print(
                f"  完成：总剖面 {stats['total']}，原质控通过 {stats['existing_qc_pass']}，"
                f"最终输出 {stats['output']}，特征缺失剔除 {stats['feature_rejected']}"
            )
        except Exception as exc:
            failures.append((doy, str(exc)))
            print(f"  失败：{exc}")

    summary = pd.DataFrame(
        [{"metric": key, "count": value} for key, value in sorted(total.items())]
    )
    range_tag = f"{args.start_doy:03d}_{args.end_doy:03d}"
    _atomic_csv(summary, args.output_dir / f"processing_summary_{range_tag}.csv")
    if failures:
        _atomic_csv(
            pd.DataFrame(failures, columns=["doy", "error"]),
            args.output_dir / f"failed_days_{range_tag}.csv",
        )
    print(f"全部完成，输出目录：{args.output_dir.resolve()}")
    print(f"累计输出 {total['output']} 条，失败日期 {len(failures)} 天")


if __name__ == "__main__":
    main()
