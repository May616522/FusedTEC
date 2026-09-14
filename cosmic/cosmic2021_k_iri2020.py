"""COSMIC-2 2021 profile QC and IRI-2020-plasmasphere k calculation.

For each profile that passes the COSMIC quality checks, this script computes

    A = integral(Ne_RO, 150 km .. h_t)
    B = integral(Ne_IRI2020, h_t .. 1336 km)
    C = integral(Ne_IRI2020, 1336 km .. 20200 km)
    k = (A + B) / (A + B + C)

The default invocation processes year 2021, day of year 056.  Input NetCDF
profiles are read directly from the daily tar.gz archive without extracting
thousands of temporary files.
"""

from __future__ import annotations

import argparse
import datetime as datetime_module
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import netCDF4
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cosmic2021_K_IRI2020"

# iri20py is kept inside the project so the calculation is reproducible and
# does not depend on the older system-wide iri2016 package.
PROJECT_IRI20PY = PROJECT_ROOT / ".deps" / "iri20py"
if PROJECT_IRI20PY.is_dir():
    sys.path.insert(0, str(PROJECT_IRI20PY))

# iri20py 0.0.5 declares Python >= 3.9 support but imports datetime.UTC,
# which was only added in Python 3.11.  Provide the equivalent UTC object for
# Python 3.9/3.10 before iri20py is imported.
if not hasattr(datetime_module, "UTC"):
    datetime_module.UTC = timezone.utc  # type: ignore[attr-defined]

try:
    from iri20py import Iri2020
    from iri20py.settings import Settings
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name == "iri20py":
        message = "缺少 IRI-2020 依赖。请在当前虚拟环境中运行：\npython -m pip install iri20py==0.0.5"
    else:
        message = f"iri20py 已安装，但缺少其依赖 {exc.name!r}。请在当前虚拟环境中安装该依赖。"
    raise SystemExit(message) from exc
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"iri20py 已安装，但导入失败：{exc}") from exc


OUTPUT_COLUMNS = ["datetime", "lon", "lat", "k", "tec_ro","tec_1336","tec_20200"]
TECU = 1.0e16


@dataclass(frozen=True)
class CosmicQcConfig:
    """Thresholds from Code/cosmic/Cosmic_process.py."""

    nm_f2_min_m3: float = 2.0e10
    nm_f2_max_m3: float = 2.0e12
    hm_f2_min_km: float = 200.0
    hm_f2_max_km: float = 450.0
    gradient_max_m4: float = -0.1e5
    md_min: float = 0.0
    md_max: float = 0.05
    delta_max: float = 0.02
    smooth_window: int = 9


@dataclass
class RoProfile:
    time: datetime
    alt_km: np.ndarray
    ne_cm3: np.ndarray
    ne_smooth_cm3: np.ndarray
    nm_f2_cm3: float
    hm_f2_km: float
    nm_f2_lat: float
    nm_f2_lon: float


def _to_float_array(variable: netCDF4.Variable) -> np.ndarray:
    values = variable[:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _profile_time(ds: netCDF4.Dataset) -> datetime:
    second = float(ds.getncattr("second")) if "second" in ds.ncattrs() else 0.0
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
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


def read_profile(file_object: BinaryIO, cfg: CosmicQcConfig) -> RoProfile:
    payload = file_object.read()
    ds = netCDF4.Dataset("in_memory.nc", mode="r", memory=payload)
    try:
        alt = _to_float_array(ds.variables["MSL_alt"])
        ne = _to_float_array(ds.variables["ELEC_dens"])
        valid = np.isfinite(alt) & np.isfinite(ne)
        alt, ne = alt[valid], ne[valid]
        if alt.size < max(30, cfg.smooth_window):
            raise ValueError("too_short")

        order = np.argsort(alt)
        alt, ne = alt[order], ne[order]
        # Duplicate altitudes make interpolation/integration ambiguous.
        unique_alt, unique_index = np.unique(alt, return_index=True)
        alt, ne = unique_alt, ne[unique_index]
        if alt.size < max(30, cfg.smooth_window):
            raise ValueError("too_short")

        ne_smooth = uniform_filter1d(ne, size=cfg.smooth_window, mode="nearest")
        return RoProfile(
            time=_profile_time(ds),
            alt_km=alt,
            ne_cm3=ne,
            ne_smooth_cm3=ne_smooth,
            nm_f2_cm3=float(ds.getncattr("edmax")),
            hm_f2_km=float(ds.getncattr("edmaxalt")),
            nm_f2_lat=float(ds.getncattr("edmaxlat")),
            nm_f2_lon=float(ds.getncattr("edmaxlon")),
        )
    finally:
        ds.close()


def quality_control(profile: RoProfile, cfg: CosmicQcConfig) -> tuple[bool, str]:
    """Apply COSMIC_process.py thresholds plus indispensable data checks.

    NmF2 is enabled here because the threshold is defined by the project QC and
    was also enabled in the user's earlier TEC script.  The old script's
    150--500 km negative-density rejection is retained; negative values are not
    silently replaced by zero.
    """

    alt = profile.alt_km
    ne = profile.ne_cm3
    smooth = profile.ne_smooth_cm3

    if alt[0] > 150.0 or alt[-1] < 490.0:
        return False, "height_coverage"

    nm_f2_m3 = profile.nm_f2_cm3 * 1.0e6
    if not (cfg.nm_f2_min_m3 <= nm_f2_m3 <= cfg.nm_f2_max_m3):
        return False, "NmF2_limit"
    if not (cfg.hm_f2_min_km <= profile.hm_f2_km <= cfg.hm_f2_max_km):
        return False, "hmF2_limit"

    core = (alt >= 150.0) & (alt <= 500.0)
    if core.sum() < 20 or np.any(ne[core] < 0.0):
        return False, "negative_core"

    ne_420 = float(np.interp(420.0, alt, smooth))
    ne_490 = float(np.interp(490.0, alt, smooth))
    gradient_m4 = (ne_490 - ne_420) * 1.0e6 / (70.0 * 1000.0)
    if not np.isfinite(gradient_m4) or gradient_m4 > cfg.gradient_max_m4:
        return False, "gradient_error"

    md_mask = alt >= 180.0
    if md_mask.sum() == 0 or np.any(smooth[md_mask] <= 0.0):
        return False, "md_invalid"
    md = float(np.mean(np.abs(ne[md_mask] - smooth[md_mask]) / smooth[md_mask]))
    if not np.isfinite(md) or not (cfg.md_min <= md <= cfg.md_max):
        return False, "md_error"

    delta_mask = alt >= 300.0
    count = int(delta_mask.sum())
    if count == 0 or profile.nm_f2_cm3 <= 0.0:
        return False, "delta_invalid"
    residual = ne[delta_mask] - smooth[delta_mask]
    delta = float(np.sqrt(np.sum(residual**2) / (count * profile.nm_f2_cm3**2)))
    if not np.isfinite(delta) or delta > cfg.delta_max:
        return False, "Delta_error"

    return True, "pass"


def _bounded_integral(
    altitude_km: np.ndarray,
    density_m3: np.ndarray,
    lower_km: float,
    upper_km: float,
) -> float:
    """Trapezoid integral with exact interpolated values at both boundaries."""

    if upper_km <= lower_km:
        return 0.0
    if altitude_km[0] > lower_km or altitude_km[-1] < upper_km:
        raise ValueError("integration_range_not_covered")
    inside = (altitude_km > lower_km) & (altitude_km < upper_km)
    x = np.concatenate(([lower_km], altitude_km[inside], [upper_km]))
    y = np.concatenate(
        (
            [np.interp(lower_km, altitude_km, density_m3)],
            density_m3[inside],
            [np.interp(upper_km, altitude_km, density_m3)],
        )
    )
    # np.trapezoid was introduced in NumPy 2.0; NumPy 1.x provides the
    # numerically equivalent np.trapz implementation.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x * 1000.0) / TECU)


def integrate_ro(profile: RoProfile) -> tuple[float, float]:
    h_top = float(profile.alt_km[-1])
    if not (150.0 < h_top < 1336.0):
        raise ValueError("invalid_profile_top")
    interval = (profile.alt_km >= 150.0) & (profile.alt_km <= h_top)
    if np.any(profile.ne_cm3[interval] < 0.0):
        raise ValueError("negative_integration_density")
    tec_ro = _bounded_integral(
        profile.alt_km,
        profile.ne_cm3 * 1.0e6,
        150.0,
        h_top,
    )
    if not np.isfinite(tec_ro) or tec_ro <= 0.0:
        raise ValueError("invalid_ro_tec")
    return tec_ro, h_top


def iri_altitude_grid(h_top: float) -> np.ndarray:
    """A deterministic non-uniform grid with exact splice/orbit boundaries."""

    sections = [
        np.arange(h_top, 2000.0, 5.0),
        np.arange(2000.0, 5000.0, 20.0),
        np.arange(5000.0, 10000.0, 50.0),
        np.arange(10000.0, 20200.0, 100.0),
        np.asarray([h_top, 1336.0, 20200.0]),
    ]
    return np.unique(np.concatenate(sections)).astype(np.float32)


def calculate_k(
    profile: RoProfile,
    tec_ro: float,
    h_top: float,
    iri: Iri2020,
    iri_settings: Settings,
) -> float:
    alt_iri = iri_altitude_grid(h_top)
    _, modeled = iri.evaluate(
        profile.time,
        profile.nm_f2_lat,
        profile.nm_f2_lon,
        alt_iri,
        iri_settings,
        tzaware=True,
    )
    ne_iri_m3 = np.asarray(modeled["Ne"].values, dtype=np.float64) * 1.0e6
    if np.any(~np.isfinite(ne_iri_m3)) or np.any(ne_iri_m3 < 0.0):
        raise ValueError("invalid_iri_density")

    tec_top = _bounded_integral(alt_iri, ne_iri_m3, h_top, 1336.0)
    tec_plas = _bounded_integral(alt_iri, ne_iri_m3, 1336.0, 20200.0)
    tec_1336 = tec_ro + tec_top
    tec_20200 = tec_1336 + tec_plas
    if tec_20200 <= 0.0:
        raise ValueError("invalid_total_tec")
    k = tec_20200/tec_1336
    
    if not ( k >= 1.0):
        raise ValueError("invalid_k")
    return float(k),float(tec_20200),float(tec_1336)


def process_daily_archive(
    archive_path: Path,
    output_path: Path,
    plasmasphere_model: str = "Ozhogin",
) -> tuple[pd.DataFrame, Counter]:
    cfg = CosmicQcConfig()
    iri = Iri2020()
    iri_settings = Settings(plasmasphere=plasmasphere_model)
    stats: Counter = Counter()
    records: list[dict[str, object]] = []

    # Streaming mode is essential: random extraction from a compressed tar is
    # quadratic for thousands of members because it repeatedly decompresses.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("_nc"):
                continue
            stats["total"] += 1
            file_object = archive.extractfile(member)
            if file_object is None:
                stats["extract_error"] += 1
                continue
            try:
                profile = read_profile(file_object, cfg)
                passed, reason = quality_control(profile, cfg)
                if not passed:
                    stats[reason] += 1
                    continue
                stats["qc_pass"] += 1
                tec_ro, h_top = integrate_ro(profile)
                k_value,tec_20200,tec_1336 = calculate_k(profile, tec_ro, h_top, iri, iri_settings)
                time_text = profile.time.isoformat(timespec="microseconds").replace(
                    "+00:00", "Z"
                )
                time_text = time_text.replace(".000000Z", "Z")
                records.append(
                    {
                        "datetime": time_text,
                        "lon": profile.nm_f2_lon,
                        "lat": profile.nm_f2_lat,
                        "k": k_value,
                        "tec_ro": tec_ro,
                        "tec_1336":tec_1336,
                        "tec_20200":tec_20200,
                    }
                )
                stats["output"] += 1
            except (KeyError, ValueError, RuntimeError, OSError) as exc:
                stats[str(exc)] += 1

            if stats["total"] % 250 == 0:
                print(
                    f"已读取 {stats['total']} 条，质控通过 {stats['qc_pass']} 条，"
                    f"已计算 {stats['output']} 条"
                )

    frame = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.10g")
    return frame, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COSMIC-2 单日质控与 IRI-2020 k 值计算")
    parser.add_argument("--year", type=int, default=2021, help="年份，默认 2021")
    parser.add_argument("--doy", type=int, default=56, help="年积日，默认 56")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plasmasphere-model",
        choices=("Ozhogin", "Gallagher"),
        default="Ozhogin",
        help="IRI-2020 等离子层模型，默认 Ozhogin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doy = f"{args.doy:03d}"
    archive = args.input_dir / f"ionPrf_prov1_{args.year}_{doy}.tar.gz"
    output = args.output_dir / f"COSMIC2_K_IRI2020_{args.year}_{doy}.csv"
    if not archive.is_file():
        raise SystemExit(f"找不到输入文件：{archive}")

    print(f"输入：{archive}")
    print(f"IRI-2020：IRICor2 + {args.plasmasphere_model}")
    frame, stats = process_daily_archive(archive, output, args.plasmasphere_model)
    print("\n处理完成")
    print(f"总剖面：{stats['total']}")
    print(f"质控通过：{stats['qc_pass']}")
    print(f"输出记录：{len(frame)}")
    print(f"结果：{output}")
    print("未通过/异常统计：")
    for reason, count in sorted(stats.items()):
        if reason not in {"total", "qc_pass", "output"}:
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
