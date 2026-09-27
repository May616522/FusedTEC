"""批量处理 COSMIC-2 ionPrf 剖面，并附加插值得到的 GIM VTEC。

脚本以流式方式读取每日 ``tar.gz`` 压缩包，调用 ``Cosmic_process.py`` 中既有
质量控制，过滤 2021 年降轨阶段，从 60 km 积分电子密度至有效剖面顶部，
再按剖面时空位置插值 IGS GIM VTEC。

积分时排除负电子密度点；若剩余有效样本的最低高度高于 60 km，则从实际
最低样本开始积分，而不是丢弃整条剖面。默认结果写入
``Data/Cosmic2021/processed_csv_gim/``。每个输入压缩包生成一个科学数据 CSV
和一个质控/审计 CSV；压缩包成员仅在内存中读取，不解压到磁盘。
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

import netCDF4
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
CODE_ROOT = PROJECT_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from helper.interpolator import GIMInterpolator  # noqa: E402

import Cosmic_process as existing_cosmic_qc  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021"
DEFAULT_GIM_DIR = PROJECT_ROOT / "Data" / "GIM2021" / "IGS"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "processed_csv_gim"

OUTPUT_COLUMNS = [
    "time",
    "lat",
    "lon",
    "tec0",
    "tec1",
    "tec_cal",
    "integration_bottom_km",
    "gim_vtec",
]
AUDIT_COLUMNS = [
    "source_archive",
    "source_member",
    "leo_id",
    "edmaxtime_gps_s",
    "time_utc",
    "original_qc_pass",
    "orbit_alt_source",
    "orbit_alt_km",
    "orbit_stage_pass",
    "integration_pass",
    "integration_bottom_km",
    "negative_density_points_removed",
    "gim_vtec",
    "final_status",
    "reject_reason",
    "detail",
]

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_MINUS_UTC_SECONDS_2021 = 18
DEFAULT_ORBIT_ALT_MIN_KM = 520.0
DEFAULT_ORBIT_ALT_MAX_KM = 570.0
MANEUVER_WINDOWS = {
    "C2E6": (date(2021, 1, 10), date(2021, 2, 3)),
    "C2E4": (date(2021, 2, 15), date(2021, 2, 20)),
}
LEO_PATTERN = re.compile(r"(?:^|[_/])(C2E[1-6])(?:\.|_)", re.IGNORECASE)


class ProfileRejected(ValueError):
    """表示预期的单剖面拒绝，并携带稳定、可审计的拒绝原因。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _finite_float(value: object, field_name: str) -> float:
    """将 NetCDF 标量转换为有限的 Python 浮点数。"""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileRejected("missing_required_field", field_name) from exc
    if not np.isfinite(result):
        raise ProfileRejected("missing_required_field", field_name)
    return result


def _variable_as_float(variable: netCDF4.Variable) -> np.ndarray:
    """读取 NetCDF 变量为浮点数组，并把掩码值转换为 NaN。"""

    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def gps_seconds_to_utc(gps_seconds: float) -> datetime:
    """把 2021 年 COSMIC-2 GPS 秒转换为 UTC 时间。"""

    seconds = _finite_float(gps_seconds, "edmaxtime")
    return GPS_EPOCH + timedelta(
        seconds=seconds - GPS_MINUS_UTC_SECONDS_2021
    )


def _calendar_time(ds: netCDF4.Dataset) -> datetime:
    """读取 ionPrf 中采用 GPS 时间尺度的日历字段并转换为 UTC。"""

    required = ("year", "month", "day", "hour", "minute", "second")
    if any(name not in ds.ncattrs() for name in required):
        raise ProfileRejected("missing_required_field", "calendar_time")

    second = _finite_float(ds.getncattr("second"), "second")
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
    if microsecond == 1_000_000:
        whole_second += 1
        microsecond = 0
    try:
        return datetime(
            int(ds.getncattr("year")),
            int(ds.getncattr("month")),
            int(ds.getncattr("day")),
            int(ds.getncattr("hour")),
            int(ds.getncattr("minute")),
            whole_second,
            microsecond,
            tzinfo=timezone.utc,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileRejected("invalid_time_or_position", "calendar_time") from exc


def _iso_utc(value: datetime) -> str:
    """把时间格式化为带 Z 后缀的 UTC ISO 字符串。"""

    text = value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return text.replace(".000000Z", "Z")


def _read_metadata_and_profile(payload: bytes) -> dict[str, object]:
    """从内存 NetCDF 中读取质控通过后续处理所需的元数据和剖面变量。"""

    try:
        ds = netCDF4.Dataset("ionPrf_in_memory.nc", mode="r", memory=payload)
    except (OSError, RuntimeError) as exc:
        raise ProfileRejected("unreadable_netcdf", str(exc)) from exc

    try:
        required_attrs = ("edmaxtime", "edmaxlat", "edmaxlon", "tec0", "tec1")
        missing_attrs = [name for name in required_attrs if name not in ds.ncattrs()]
        missing_vars = [
            name for name in ("MSL_alt", "ELEC_dens") if name not in ds.variables
        ]
        if missing_attrs or missing_vars:
            detail = ",".join(missing_attrs + missing_vars)
            raise ProfileRejected("missing_required_field", detail)

        gps_seconds = _finite_float(ds.getncattr("edmaxtime"), "edmaxtime")
        time_utc = gps_seconds_to_utc(gps_seconds)
        gps_calendar = GPS_EPOCH + timedelta(seconds=gps_seconds)
        calendar = _calendar_time(ds)
        if abs((calendar - gps_calendar).total_seconds()) > 1.0:
            raise ProfileRejected(
                "invalid_time_or_position",
                "edmaxtime/calendar mismatch",
            )

        lat = _finite_float(ds.getncattr("edmaxlat"), "edmaxlat")
        lon = _finite_float(ds.getncattr("edmaxlon"), "edmaxlon")
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise ProfileRejected("invalid_time_or_position", "lat/lon out of range")

        alt_km = _variable_as_float(ds.variables["MSL_alt"])
        ne_cm3 = _variable_as_float(ds.variables["ELEC_dens"])
        if alt_km.size != ne_cm3.size:
            raise ProfileRejected("missing_required_field", "profile length mismatch")

        attrs = {
            name: ds.getncattr(name)
            for name in ("edorbalt", "topalt")
            if name in ds.ncattrs()
        }
        return {
            "gps_seconds": gps_seconds,
            "time_utc": time_utc,
            "lat": lat,
            "lon": lon,
            "tec0": _finite_float(ds.getncattr("tec0"), "tec0"),
            "tec1": _finite_float(ds.getncattr("tec1"), "tec1"),
            "alt_km": alt_km,
            "ne_cm3": ne_cm3,
            "attrs": attrs,
        }
    finally:
        ds.close()


def _run_existing_qc(payload: bytes) -> tuple[bool, str]:
    """原样调用 ``Cosmic_process.py`` 中的既有质控逻辑。"""

    try:
        profile = existing_cosmic_qc.read_ro_profile(io.BytesIO(payload))
        return existing_cosmic_qc.quality_control(
            profile, existing_cosmic_qc.CosmicConfig()
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProfileRejected("failed_existing_qc", str(exc)) from exc


def parse_leo_id(member_name: str) -> str:
    """从压缩包成员名解析 COSMIC-2 低轨卫星编号。"""

    match = LEO_PATTERN.search(member_name.replace("\\", "/"))
    if match is None:
        raise ProfileRejected("missing_required_field", "leo_id")
    return match.group(1).upper()


def _member_is_safe(member: tarfile.TarInfo) -> bool:
    """确认压缩包成员是安全的普通文件，避免路径穿越。"""

    path = PurePosixPath(member.name.replace("\\", "/"))
    return member.isfile() and not path.is_absolute() and ".." not in path.parts


def in_maneuver_window(leo_id: str, time_utc: datetime) -> bool:
    """判断卫星在给定时刻是否处于预设降轨机动时间窗。"""

    window = MANEUVER_WINDOWS.get(leo_id)
    return window is not None and window[0] <= time_utc.date() <= window[1]


def choose_orbit_altitude(
    attrs: dict[str, object], altitude_km: np.ndarray
) -> tuple[float, str]:
    """依次选用 edorbalt、topalt 或有效剖面最大高度作为轨道高度。"""

    for name in ("edorbalt", "topalt"):
        try:
            value = float(attrs[name])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        # -999 等 NetCDF 缺测标记虽然是有限数，却不是有效物理高度，需继续回退。
        if np.isfinite(value) and value > 0.0:
            return value, name

    finite_altitude = altitude_km[np.isfinite(altitude_km)]
    if finite_altitude.size:
        return float(np.max(finite_altitude)), "max_MSL_alt"
    raise ProfileRejected("missing_required_field", "orbit_altitude")


def integrate_electron_density(
    altitude_km: np.ndarray, density_cm3: np.ndarray
) -> tuple[float, float, int]:
    """从 60 km 或有效剖面底部开始积分非负电子密度。

    返回 ``(tec_cal, integration_bottom_km, removed_negative_count)``；若第一个
    可用样本高于 60 km，则不向下外推。
    """

    finite_pair = np.isfinite(altitude_km) & np.isfinite(density_cm3)
    removed_negative_count = int(np.sum(finite_pair & (density_cm3 < 0.0)))
    valid = finite_pair & (density_cm3 >= 0.0)
    altitude = np.asarray(altitude_km[valid], dtype=np.float64)
    density = np.asarray(density_cm3[valid], dtype=np.float64)
    if altitude.size < 2:
        raise ProfileRejected("insufficient_integration_points")

    order = np.argsort(altitude, kind="stable")
    altitude, density = altitude[order], density[order]

    # 仅当高度和密度完全相同时删除重复点，避免误合并同高度的不同观测。
    unique_altitude: list[float] = []
    unique_density: list[float] = []
    start = 0
    while start < altitude.size:
        end = start + 1
        while end < altitude.size and altitude[end] == altitude[start]:
            end += 1
        same_height_density = density[start:end]
        if not np.all(same_height_density == same_height_density[0]):
            raise ProfileRejected(
                "non_monotonic_or_duplicate_altitude",
                f"conflicting density at {altitude[start]:.6g} km",
            )
        unique_altitude.append(float(altitude[start]))
        unique_density.append(float(same_height_density[0]))
        start = end

    altitude = np.asarray(unique_altitude)
    density = np.asarray(unique_density)
    if altitude[-1] <= 60.0:
        raise ProfileRejected("profile_does_not_cover_60km")

    integration_bottom_km = max(60.0, float(altitude[0]))
    at_or_above = altitude >= integration_bottom_km
    x = altitude[at_or_above]
    y = density[at_or_above]
    if integration_bottom_km == 60.0 and (x.size == 0 or x[0] > 60.0):
        value_at_60 = float(np.interp(60.0, altitude, density))
        x = np.concatenate(([60.0], x))
        y = np.concatenate(([value_at_60], y))
    if x.size < 2:
        raise ProfileRejected("insufficient_integration_points")

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # 兼容 NumPy 1.x
        trapezoid = np.trapz
    tec = float(trapezoid(y, x) * 1.0e-7)
    if not np.isfinite(tec):
        raise ProfileRejected("integration_failed", "non-finite tec_cal")
    return tec, integration_bottom_km, removed_negative_count


def _new_audit_row(archive_name: str, member_name: str) -> dict[str, object]:
    """创建带默认值的单剖面审计记录。"""

    return {
        "source_archive": archive_name,
        "source_member": member_name,
        "leo_id": "",
        "edmaxtime_gps_s": np.nan,
        "time_utc": "",
        "original_qc_pass": False,
        "orbit_alt_source": "",
        "orbit_alt_km": np.nan,
        "orbit_stage_pass": False,
        "integration_pass": False,
        "integration_bottom_km": np.nan,
        "negative_density_points_removed": 0,
        "gim_vtec": np.nan,
        "final_status": "rejected",
        "reject_reason": "",
        "detail": "",
    }


def _reject(
    audit: dict[str, object], reason: str, detail: str, stats: Counter
) -> None:
    """在审计记录中登记剖面拒绝原因并累计相应计数。"""

    audit["reject_reason"] = reason
    audit["detail"] = detail.replace("\r", " ").replace("\n", " ")[:500]
    stats[reason] += 1


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    """先写临时文件再原子替换目标 CSV，避免留下半写入结果。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )
    os.replace(temporary, destination)


def process_daily_archive(
    archive_path: Path,
    output_dir: Path,
    gim: GIMInterpolator,
    *,
    orbit_alt_min_km: float = DEFAULT_ORBIT_ALT_MIN_KM,
    orbit_alt_max_km: float = DEFAULT_ORBIT_ALT_MAX_KM,
    max_profiles: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """处理一个每日压缩包，并原子写出科学数据与审计 CSV。

    每条剖面依次经过文件安全检查、既有质控、降轨筛选、电子密度积分和
    GIM 插值；单文件异常只记录到审计表，不会导致整日数据丢失。
    """

    records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    stats: Counter = Counter()
    seen_members: set[str] = set()

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            normalized_name = member.name.replace("\\", "/")
            if not normalized_name.startswith("ionPrf_") or not normalized_name.endswith(
                "_nc"
            ):
                continue
            stats["total"] += 1
            audit = _new_audit_row(archive_path.name, normalized_name)

            if not _member_is_safe(member):
                _reject(audit, "unsafe_archive_member", "", stats)
                audits.append(audit)
                continue
            if normalized_name in seen_members:
                _reject(audit, "duplicate_event", "duplicate member name", stats)
                audits.append(audit)
                continue
            seen_members.add(normalized_name)

            file_object: BinaryIO | None = archive.extractfile(member)
            if file_object is None:
                _reject(audit, "unreadable_netcdf", "extractfile returned None", stats)
                audits.append(audit)
                continue

            try:
                payload = file_object.read()
                metadata = _read_metadata_and_profile(payload)
                time_utc = metadata["time_utc"]
                assert isinstance(time_utc, datetime)
                audit["edmaxtime_gps_s"] = metadata["gps_seconds"]
                audit["time_utc"] = _iso_utc(time_utc)

                leo_id = parse_leo_id(normalized_name)
                audit["leo_id"] = leo_id

                passed, qc_reason = _run_existing_qc(payload)
                audit["original_qc_pass"] = passed
                if not passed:
                    raise ProfileRejected("failed_existing_qc", qc_reason)
                stats["original_qc_pass"] += 1

                if in_maneuver_window(leo_id, time_utc):
                    raise ProfileRejected("official_maneuver_window")

                altitude = metadata["alt_km"]
                density = metadata["ne_cm3"]
                assert isinstance(altitude, np.ndarray)
                assert isinstance(density, np.ndarray)
                orbit_alt, orbit_source = choose_orbit_altitude(
                    metadata["attrs"], altitude  # type: ignore[arg-type]
                )
                audit["orbit_alt_source"] = orbit_source
                audit["orbit_alt_km"] = orbit_alt
                if not orbit_alt_min_km <= orbit_alt <= orbit_alt_max_km:
                    raise ProfileRejected(
                        "orbit_alt_outside_configured_range",
                        f"{orbit_alt:.6g} km not in "
                        f"[{orbit_alt_min_km:.6g}, {orbit_alt_max_km:.6g}] km",
                    )
                audit["orbit_stage_pass"] = True
                stats["orbit_stage_pass"] += 1

                (
                    tec_cal,
                    integration_bottom_km,
                    removed_negative_count,
                ) = integrate_electron_density(altitude, density)
                audit["integration_pass"] = True
                audit["integration_bottom_km"] = integration_bottom_km
                audit["negative_density_points_removed"] = removed_negative_count
                stats["integration_pass"] += 1
                stats["negative_density_points_removed"] += removed_negative_count

                try:
                    gim_vtec = float(
                        gim.query(time_utc, metadata["lat"], metadata["lon"])
                    )
                except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    raise ProfileRejected("gim_interpolation_failed", str(exc)) from exc
                if not np.isfinite(gim_vtec):
                    raise ProfileRejected("gim_interpolation_failed", "non-finite GIM VTEC")

                audit["gim_vtec"] = gim_vtec
                audit["final_status"] = "passed"
                records.append(
                    {
                        "time": audit["time_utc"],
                        "lat": metadata["lat"],
                        "lon": metadata["lon"],
                        "tec0": metadata["tec0"],
                        "tec1": metadata["tec1"],
                        "tec_cal": tec_cal,
                        "integration_bottom_km": integration_bottom_km,
                        "gim_vtec": gim_vtec,
                    }
                )
                stats["output"] += 1
            except ProfileRejected as exc:
                _reject(audit, exc.reason, exc.detail, stats)
            except Exception as exc:  # 单个畸形剖面不应导致整日处理失败
                _reject(audit, "unreadable_netcdf", str(exc), stats)
            finally:
                file_object.close()
                audits.append(audit)

            if stats["total"] % 250 == 0:
                print(
                    f"  已读取 {stats['total']} 条，原 QC 通过 "
                    f"{stats['original_qc_pass']} 条，轨道通过 "
                    f"{stats['orbit_stage_pass']} 条，积分通过 "
                    f"{stats['integration_pass']} 条，输出 {stats['output']} 条"
                )
            if max_profiles is not None and stats["total"] >= max_profiles:
                break

    science = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if not science.empty:
        science = science.sort_values("time", kind="stable").reset_index(drop=True)
    audit_frame = pd.DataFrame.from_records(audits, columns=AUDIT_COLUMNS)

    match = re.search(r"_(\d{4})_(\d{3})\.tar\.gz$", archive_path.name)
    stem = (
        f"cosmic2_ionPrf_{match.group(1)}_{match.group(2)}"
        if match
        else archive_path.name.removesuffix(".tar.gz")
    )
    _atomic_csv(science, output_dir / f"{stem}.csv")
    _atomic_csv(audit_frame, output_dir / f"{stem}_qc_log.csv")
    return science, audit_frame, stats


def _selected_archives(
    input_dir: Path, year: int, start_doy: int, end_doy: int
) -> Iterable[tuple[int, Path]]:
    """按年份和年积日范围依次生成每日 ionPrf 压缩包路径。"""

    for doy in range(start_doy, end_doy + 1):
        yield doy, input_dir / f"ionPrf_prov1_{year}_{doy:03d}.tar.gz"


def parse_args() -> argparse.Namespace:
    """定义并解析批处理命令行参数。"""

    parser = argparse.ArgumentParser(
        description="COSMIC-2 2021 ionPrf 批处理、TEC 积分与 GIM 时空插值"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--gim-dir", type=Path, default=DEFAULT_GIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--start-doy", type=int, default=1)
    parser.add_argument("--end-doy", type=int, default=365)
    parser.add_argument(
        "--orbit-alt-min",
        type=float,
        default=DEFAULT_ORBIT_ALT_MIN_KM,
        help=f"轨道高度下限 km，默认 {DEFAULT_ORBIT_ALT_MIN_KM:g}",
    )
    parser.add_argument(
        "--orbit-alt-max",
        type=float,
        default=DEFAULT_ORBIT_ALT_MAX_KM,
        help=f"轨道高度上限 km，默认 {DEFAULT_ORBIT_ALT_MAX_KM:g}",
    )
    return parser.parse_args()


def main() -> None:
    """逐日创建 GIM 插值器，处理 ionPrf 压缩包并打印汇总状态。"""

    args = parse_args()
    if args.year != 2021:
        raise SystemExit("本脚本的 GPS-UTC 与轨道窗口规则仅适用于 2021 年")
    if not (1 <= args.start_doy <= args.end_doy <= 365):
        raise SystemExit("DOY 范围必须满足 1 <= start-doy <= end-doy <= 365")
    if not 0.0 < args.orbit_alt_min < args.orbit_alt_max:
        raise SystemExit("轨道高度范围必须满足 0 < orbit-alt-min < orbit-alt-max")
    if not args.input_dir.is_dir():
        raise SystemExit(f"COSMIC 输入目录不存在：{args.input_dir}")
    if not args.gim_dir.is_dir():
        raise SystemExit(f"GIM 输入目录不存在：{args.gim_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed_days: list[tuple[int, str]] = []
    total_stats: Counter = Counter()
    selected = list(
        _selected_archives(args.input_dir, args.year, args.start_doy, args.end_doy)
    )
    print(f"将处理 {len(selected)} 天")
    print(f"COSMIC：{args.input_dir.resolve()}")
    print(f"GIM：{args.gim_dir.resolve()}")
    print(f"输出：{args.output_dir.resolve()}")
    print(f"轨道高度范围：{args.orbit_alt_min:g}–{args.orbit_alt_max:g} km")

    for index, (doy, archive_path) in enumerate(selected, start=1):
        if not archive_path.is_file():
            message = f"输入文件不存在：{archive_path.name}"
            failed_days.append((doy, message))
            print(f"[{index}/{len(selected)}] DOY {doy:03d} 跳过：{message}")
            continue

        print(f"[{index}/{len(selected)}] DOY {doy:03d}：{archive_path.name}")
        try:
            # 每日新建插值器，防止全年运行时 IONEX 缓存长期占用整年内存。
            gim = GIMInterpolator(args.gim_dir)
            science, _, stats = process_daily_archive(
                archive_path,
                args.output_dir,
                gim,
                orbit_alt_min_km=args.orbit_alt_min,
                orbit_alt_max_km=args.orbit_alt_max,
            )
            total_stats.update(stats)
            print(f"  完成：{len(science)} 条记录")
        except (OSError, RuntimeError, tarfile.TarError, ValueError) as exc:
            failed_days.append((doy, str(exc)))
            print(f"  当日失败：{exc}")

    print("\n全年/所选时段汇总")
    print(f"总剖面：{total_stats['total']}")
    print(f"原 QC 通过：{total_stats['original_qc_pass']}")
    print(f"轨道筛选通过：{total_stats['orbit_stage_pass']}")
    print(f"积分通过：{total_stats['integration_pass']}")
    print(f"输出记录：{total_stats['output']}")
    if failed_days:
        print(f"失败/缺失日期：{len(failed_days)}")
        for doy, message in failed_days:
            print(f"  {doy:03d}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
