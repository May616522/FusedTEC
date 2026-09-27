"""精简 COSMIC-2 建模 CSV，并删除逐剖面质量控制日志。

处理 ``Data/Cosmic2021/modeling_csv_qc`` 中所有严格符合年份/年积日命名的
科学 CSV：新增 ``residual = gim_tec - tec0``，保留 ``gim_vtec``，删除重复的
``gim_tec``，并删除 ``land_fraction`` 与 ``surface_type``，保留数值型
``is_land``。所有科学 CSV 成功改写后，再删除 ``*_qc_log.csv``。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "modeling_csv_qc"
SCIENCE_PATTERN = re.compile(r"^cosmic2_model_(\d{4})_(\d{3})\.csv$")
QC_LOG_PATTERN = re.compile(r"^cosmic2_model_(\d{4})_(\d{3})_qc_log\.csv$")


def discover_files(input_dir: Path, year: int) -> tuple[list[Path], list[Path]]:
    """发现严格匹配指定年份的日级科学 CSV 和质控日志。"""

    science: list[Path] = []
    qc_logs: list[Path] = []
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        science_match = SCIENCE_PATTERN.match(path.name)
        if science_match and int(science_match.group(1)) == year:
            science.append(path)
            continue
        log_match = QC_LOG_PATTERN.match(path.name)
        if log_match and int(log_match.group(1)) == year:
            qc_logs.append(path)
    return sorted(science), sorted(qc_logs)


def validate_science_file(path: Path) -> None:
    """在改写前确认计算 residual 所需字段存在且 GIM 重复列一致。"""

    frame = pd.read_csv(path)
    required = {"tec0", "gim_vtec"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
    if "gim_tec" in frame.columns:
        same = np.allclose(
            frame["gim_tec"].to_numpy(dtype=float),
            frame["gim_vtec"].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        if not same:
            raise ValueError(f"{path.name} 的 gim_tec 与 gim_vtec 不一致，停止处理")


def simplify_science_file(path: Path, force: bool = False) -> tuple[int, bool]:
    """原子改写单个科学 CSV，返回记录条数和是否实际发生改写。"""

    frame = pd.read_csv(path)
    gim_source = "gim_tec" if "gim_tec" in frame.columns else "gim_vtec"
    expected_residual = (
        pd.to_numeric(frame[gim_source], errors="coerce")
        - pd.to_numeric(frame["tec0"], errors="coerce")
    )
    redundant_columns = {"gim_tec", "land_fraction", "surface_type"}
    already_simplified = (
        "residual" in frame.columns
        and redundant_columns.isdisjoint(frame.columns)
        and np.allclose(
            pd.to_numeric(frame["residual"], errors="coerce").to_numpy(dtype=float),
            expected_residual.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
    )
    if already_simplified and not force:
        return len(frame), False

    # residual 放在 tec0 后面，目标值和观测值相邻，便于后续建模检查。
    frame = frame.drop(
        columns=["residual", "gim_tec", "land_fraction", "surface_type"],
        errors="ignore",
    )
    insert_at = frame.columns.get_loc("tec0") + 1
    frame.insert(insert_at, "residual", expected_residual)

    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.15g",
    )
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return len(frame), True


def process_directory(
    input_dir: Path, year: int, dry_run: bool = False, force: bool = False
) -> None:
    """先验证并改写全部科学文件，全部成功后删除质控日志。"""

    science, qc_logs = discover_files(input_dir, year)
    if not science:
        raise SystemExit(f"没有找到 {year} 年日级科学 CSV：{input_dir}")

    expected_doys = set(range(1, 366))
    actual_doys = {int(SCIENCE_PATTERN.match(path.name).group(2)) for path in science}
    missing_doys = sorted(expected_doys.difference(actual_doys))
    if missing_doys:
        raise ValueError(f"科学 CSV 缺少年积日：{missing_doys}")

    print(f"发现科学 CSV {len(science)} 个，质控日志 {len(qc_logs)} 个")
    for path in science:
        validate_science_file(path)
    print("全部科学 CSV 预检查通过")
    if dry_run:
        print("dry-run：未改写 CSV，也未删除质控日志")
        return

    total_rows = 0
    changed_files = 0
    locked_files: list[Path] = []
    for index, path in enumerate(science, start=1):
        try:
            rows, changed = simplify_science_file(path, force=force)
            total_rows += rows
            changed_files += int(changed)
        except PermissionError:
            locked_files.append(path)
        if index % 50 == 0 or index == len(science):
            print(f"已改写 {index}/{len(science)} 个科学 CSV")

    if locked_files:
        print("以下文件正被其他程序占用，本次未改写：")
        for path in locked_files:
            print(f"  {path}")
        print("质控日志尚未删除；关闭占用文件后重新运行本脚本即可补处理。")
        raise SystemExit(2)

    # 删除范围仅限经过严格正则匹配的日级质控日志。
    for path in qc_logs:
        path.unlink()
    print(
        f"完成：检查 {len(science)} 个文件、实际改写 {changed_files} 个、"
        f"共 {total_rows} 行"
    )
    print(f"已删除质控日志 {len(qc_logs)} 个")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量精简 COSMIC 建模 CSV")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不写入或删除")
    parser.add_argument("--force", action="store_true", help="强制重新写入已精简的文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{args.input_dir}")
    process_directory(args.input_dir, args.year, args.dry_run, args.force)


if __name__ == "__main__":
    main()
