"""校验并汇总 ``build_cosmic2021_model_data.py`` 生成的全年结果。"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_cosmic2021_model_data import AUDIT_COLUMNS, OUTPUT_COLUMNS


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "modeling_csv_qc"
SCIENCE_PATTERN = re.compile(r"^cosmic2_model_(\d{4})_(\d{3})\.csv$")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    """先写临时文件再替换正式汇总表。"""

    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, destination)


def _true_count(series: pd.Series) -> int:
    """兼容布尔型和文本型 CSV 字段，统计真值数量。"""

    return int(series.astype(str).str.lower().eq("true").sum())


def validate_directory(input_dir: Path, year: int) -> dict[str, object]:
    """逐日检查列结构、数值完整性和科学表/审计表对应关系。"""

    science_paths: dict[int, Path] = {}
    for path in input_dir.glob(f"cosmic2_model_{year}_???.csv"):
        match = SCIENCE_PATTERN.match(path.name)
        if match:
            science_paths[int(match.group(2))] = path
    expected_doys = set(range(1, 366))
    missing_doys = sorted(expected_doys.difference(science_paths))
    extra_doys = sorted(set(science_paths).difference(expected_doys))
    if missing_doys or extra_doys:
        raise ValueError(f"DOY 不连续：缺失={missing_doys}，额外={extra_doys}")

    daily_rows: list[dict[str, object]] = []
    rejection_counts: Counter = Counter()
    total_audit = 0
    total_qc_pass = 0
    total_orbit_pass = 0
    total_integration_pass = 0
    total_output = 0
    total_land = 0
    total_ocean = 0
    total_duplicates = 0
    maximum_cycle_error = 0.0
    maximum_gim_alias_error = 0.0

    numeric_columns = [
        name
        for name in OUTPUT_COLUMNS
        if name not in {"datetime", "surface_type"}
    ]
    cycle_pairs = [
        ("lon_s", "lon_c"),
        ("mag_lat_s", "mag_lat_c"),
        ("mag_lon_s", "mag_lon_c"),
        ("DOY_s", "DOY_c"),
        ("local_time_s", "local_time_c"),
    ]

    for doy in range(1, 366):
        science_path = science_paths[doy]
        audit_path = science_path.with_name(science_path.stem + "_qc_log.csv")
        if not audit_path.is_file():
            raise ValueError(f"DOY {doy:03d} 缺少质控日志：{audit_path.name}")
        science = pd.read_csv(science_path)
        audit = pd.read_csv(audit_path)
        if science.columns.tolist() != OUTPUT_COLUMNS:
            raise ValueError(f"{science_path.name} 列结构不一致")
        if audit.columns.tolist() != AUDIT_COLUMNS:
            raise ValueError(f"{audit_path.name} 列结构不一致")
        if science["datetime"].isna().any() or pd.to_datetime(
            science["datetime"], utc=True, errors="coerce"
        ).isna().any():
            raise ValueError(f"{science_path.name} 包含无效时间")
        if not science.empty:
            values = science[numeric_columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{science_path.name} 必需数值列包含 NaN/Inf")
            if not science["surface_type"].isin(["land", "ocean"]).all():
                raise ValueError(f"{science_path.name} 包含未知陆海类型")
            expected_land = science["is_land"].to_numpy(dtype=int) == 1
            if not np.array_equal(
                expected_land, science["surface_type"].eq("land").to_numpy()
            ):
                raise ValueError(f"{science_path.name} is_land 与 surface_type 不一致")
            if not science["lon"].between(-180.0, 180.0).all():
                raise ValueError(f"{science_path.name} 地理经度越界")
            if not science["mag_lat"].between(-90.0, 90.0).all():
                raise ValueError(f"{science_path.name} 地磁纬度越界")
            for sine, cosine in cycle_pairs:
                error = np.max(
                    np.abs(
                        science[sine].to_numpy(dtype=float) ** 2
                        + science[cosine].to_numpy(dtype=float) ** 2
                        - 1.0
                    )
                )
                maximum_cycle_error = max(maximum_cycle_error, float(error))
            alias_error = np.max(
                np.abs(
                    science["gim_vtec"].to_numpy(dtype=float)
                    - science["gim_tec"].to_numpy(dtype=float)
                )
            )
            maximum_gim_alias_error = max(maximum_gim_alias_error, float(alias_error))

        passed = int(audit["final_status"].eq("passed").sum())
        if passed != len(science):
            raise ValueError(
                f"DOY {doy:03d} 最终通过数 {passed} 与输出行数 {len(science)} 不符"
            )
        rejected = audit.loc[audit["final_status"].ne("passed"), "reject_reason"]
        rejection_counts.update(rejected.fillna("未注明").astype(str))
        duplicates = int(science.duplicated(["datetime", "lat", "lon"]).sum())
        land_count = int(science["is_land"].eq(1).sum())
        ocean_count = int(science["is_land"].eq(0).sum())

        daily_rows.append(
            {
                "doy": doy,
                "input_profiles": len(audit),
                "existing_qc_pass": _true_count(audit["existing_qc_pass"]),
                "orbit_stage_pass": _true_count(audit["orbit_stage_pass"]),
                "integration_pass": _true_count(audit["integration_pass"]),
                "output_rows": len(science),
                "land_rows": land_count,
                "ocean_rows": ocean_count,
                "duplicate_rows": duplicates,
            }
        )
        total_audit += len(audit)
        total_qc_pass += _true_count(audit["existing_qc_pass"])
        total_orbit_pass += _true_count(audit["orbit_stage_pass"])
        total_integration_pass += _true_count(audit["integration_pass"])
        total_output += len(science)
        total_land += land_count
        total_ocean += ocean_count
        total_duplicates += duplicates

    daily = pd.DataFrame(daily_rows)
    rejection = pd.DataFrame(
        [
            {"reject_reason": reason, "count": count}
            for reason, count in rejection_counts.most_common()
        ]
    )
    summary_items = [
        ("science_days", 365),
        ("audit_days", 365),
        ("input_profiles", total_audit),
        ("existing_qc_pass", total_qc_pass),
        ("orbit_stage_pass", total_orbit_pass),
        ("integration_pass", total_integration_pass),
        ("output_rows", total_output),
        ("land_rows", total_land),
        ("ocean_rows", total_ocean),
        ("duplicate_rows", total_duplicates),
        ("maximum_cycle_identity_error", maximum_cycle_error),
        ("maximum_gim_alias_error", maximum_gim_alias_error),
    ]
    summary = pd.DataFrame(summary_items, columns=["metric", "value"])
    _atomic_csv(daily, input_dir / "daily_processing_summary.csv")
    _atomic_csv(rejection, input_dir / "qc_rejection_summary.csv")
    _atomic_csv(summary, input_dir / "processing_summary.csv")
    return {
        "summary": summary,
        "daily": daily,
        "rejection": rejection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验并汇总 COSMIC 建模 CSV")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--year", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_directory(args.input_dir, args.year)
    print(result["summary"].to_string(index=False))
    print("\n质控拒绝原因：")
    print(result["rejection"].to_string(index=False))
    print(f"\n校验通过：{args.input_dir.resolve()}")


if __name__ == "__main__":
    main()
