"""批量转换 COSMIC-2 podTc2 数据，并检验 tec0 加 POD 顶部 VTEC 与 GIM 的一致性。

脚本先为每个同时具备 POD 压缩包和掩星 CSV 的日期调用转换程序，生成带
POD 顶部 VTEC 的 CSV；转换全部完成后，可继续调用分析程序汇总多日结果。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_POD_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "POD"
DEFAULT_RO_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim"
DEFAULT_POD_CSV_DIR = PROJECT_ROOT / "Data" / "Cosmic2021" / "processed_csv_gim_pod"
DEFAULT_DIAGNOSTIC_ROOT = PROJECT_ROOT / "Results" / "Cosmic2_POD_VTEC"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "Results" / "Cosmic2_POD_GIM_analysis"
ARCHIVE_PATTERN = re.compile(r"podTc2_nrt_(\d{4})_(\d{3})\.tar\.gz$")


@dataclass(frozen=True)
class BatchItem:
    """记录一个待处理日期的输入、输出路径及日期信息。"""

    year: int
    doy: int
    archive: Path
    ro_csv: Path
    output_csv: Path


def parse_doys(value: str | None) -> set[int] | None:
    """把逗号分隔的年积日参数转换为整数集合。"""

    if value is None:
        return None
    try:
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as error:
        raise argparse.ArgumentTypeError("DOY必须是逗号分隔的整数") from error


def discover_batch(pod_dir: Path, ro_dir: Path, output_dir: Path, doys: set[int] | None) -> list[BatchItem]:
    """发现 POD 与掩星 CSV 均存在的日期，并构造批处理任务。"""

    items: list[BatchItem] = []
    for archive in sorted(pod_dir.glob("podTc2_nrt_*.tar.gz")):
        match = ARCHIVE_PATTERN.match(archive.name)
        if match is None:
            continue
        year, doy = map(int, match.groups())
        if doys is not None and doy not in doys:
            continue
        ro_csv = ro_dir / f"cosmic2_ionPrF_{year}_{doy:03d}.csv"
        if not ro_csv.is_file():
            print(f"跳过DOY {doy:03d}：缺少 {ro_csv}")
            continue
        output_csv = output_dir / f"cosmic2_ionPrF_{year}_{doy:03d}_pod.csv"
        items.append(BatchItem(year, doy, archive, ro_csv, output_csv))
    if not items:
        raise SystemExit("没有找到同时具有POD压缩包和tec0/GIM CSV的日期")
    return items


def run_command(command: list[str]) -> None:
    """运行一个子脚本命令；任一命令失败时立即终止批处理。"""

    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """定义并解析批处理命令行参数。"""

    parser = argparse.ArgumentParser(description="批量完成POD补足及tec0+POD与GIM验证")
    parser.add_argument("--pod-dir", type=Path, default=DEFAULT_POD_DIR)
    parser.add_argument("--ro-dir", type=Path, default=DEFAULT_RO_DIR)
    parser.add_argument("--pod-csv-dir", type=Path, default=DEFAULT_POD_CSV_DIR)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--doys", default=None, help="逗号分隔的DOY；默认处理POD目录内全部日期")
    parser.add_argument("--force", action="store_true", help="重新生成已经存在的POD校正CSV")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    """逐日执行 POD 转换，并按需启动跨日期综合分析。"""

    args = parse_args()
    selected_doys = parse_doys(args.doys)
    args.pod_csv_dir.mkdir(parents=True, exist_ok=True)
    items = discover_batch(args.pod_dir, args.ro_dir, args.pod_csv_dir, selected_doys)
    converter = SCRIPT_PATH.with_name("add_pod_vtec_to_cosmic_csv.py")
    analyzer = SCRIPT_PATH.with_name("analyze_pod_vtec_gim.py")

    for index, item in enumerate(items, start=1):
        if item.output_csv.is_file() and not args.force:
            print(f"[{index}/{len(items)}] DOY {item.doy:03d} 已存在，跳过转换")
            continue
        print(f"[{index}/{len(items)}] 处理DOY {item.doy:03d}")
        run_command(
            [
                sys.executable,
                str(converter),
                "--pod-archive",
                str(item.archive),
                "--ro-csv",
                str(item.ro_csv),
                "--output-csv",
                str(item.output_csv),
                "--diagnostic-dir",
                str(args.diagnostic_root / f"DOY{item.doy:03d}"),
                "--year",
                str(item.year),
                "--doy",
                str(item.doy),
            ]
        )

    if args.skip_analysis:
        return
    doys_text = ",".join(f"{item.doy:03d}" for item in items)
    period_name = (
        f"DOY{items[0].doy:03d}"
        if len(items) == 1
        else f"DOY{items[0].doy:03d}-{items[-1].doy:03d}_{len(items)}days"
    )
    analysis_dir = args.analysis_root / period_name
    run_command(
        [
            sys.executable,
            str(analyzer),
            "--input-dir",
            str(args.pod_csv_dir),
            "--doys",
            doys_text,
            "--output-dir",
            str(analysis_dir),
        ]
    )
    print(f"十天分析完成：{analysis_dir.resolve()}")


if __name__ == "__main__":
    main()
