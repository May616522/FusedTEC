"""检查 COSMIC-2 podTc2 文件，并将其与 ionPrf 掩星剖面进行匹配。

程序接受已解压的 NetCDF 文件、目录或每日 ``.tar.gz`` 压缩包；压缩包直接
在内存中读取，不解压到磁盘。匹配条件为低轨卫星编号、GNSS 星座/PRN 一致，
且 GPS 时间范围重叠；不会仅按文件名匹配，也不假设两类文件是一一对应。
"""

from __future__ import annotations

import argparse
import re
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "Cosmic2_podTc2_explorer"

POD_NAME = re.compile(
    r"podTc2_(?P<leo>[^.]+)\.(?P<year>\d{4})\.(?P<doy>\d{3})\."
    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<duration>\d+)\."
    r"(?P<gnss>[A-Z]\d+)(?:\.(?P<antenna>\d+))?_",
    re.IGNORECASE,
)
ION_NAME = re.compile(
    r"ionPrf_(?P<leo>[^.]+)\.(?P<year>\d{4})\.(?P<doy>\d{3})\."
    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<gnss>[A-Z]\d+)_",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NcItem:
    """统一表示磁盘文件或压缩包内存成员中的一个 NetCDF 数据项。"""

    archive: str
    name: str
    path: Path | None = None
    content: bytes | None = None


def _is_netcdf_name(name: str, prefix: str) -> bool:
    """判断文件名是否符合指定产品前缀和 NetCDF 后缀。"""

    base = Path(name).name.lower()
    return base.startswith(prefix.lower() + "_") and (
        base.endswith("_nc") or base.endswith(".nc")
    )


def _candidate_files(source: Path, prefix: str) -> list[Path]:
    """从单文件或目录中发现压缩包及已解压的候选文件。"""

    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"路径不存在：{source}")
    archives = sorted(source.rglob(f"{prefix}*.tar.gz"))
    extracted = sorted(
        path
        for path in source.rglob(f"{prefix}*")
        if path.is_file() and _is_netcdf_name(path.name, prefix)
    )
    return archives + extracted


def iter_nc_items(source: Path, prefix: str) -> Iterator[NcItem]:
    """逐个产生候选 NetCDF；压缩包成员只读入内存而不落盘。"""

    for path in _candidate_files(source, prefix):
        if path.name.lower().endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if not member.isfile() or not _is_netcdf_name(member.name, prefix):
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        yield NcItem(str(path.resolve()), member.name, content=extracted.read())
        else:
            yield NcItem(str(path.resolve()), path.name, path=path)


@contextmanager
def open_item(item: NcItem) -> Iterator[Dataset]:
    """以统一上下文管理器打开磁盘或内存中的 NetCDF 数据集。"""

    dataset = (
        Dataset(str(item.path), "r")
        if item.path is not None
        else Dataset(item.name, "r", memory=item.content)
    )
    try:
        yield dataset
    finally:
        dataset.close()


def _scalar(value: object, default: object = np.nan) -> object:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00 ")
    array = np.asanyarray(value)
    if array.size == 0:
        return default
    value = array.reshape(-1)[0]
    return value.item() if hasattr(value, "item") else value


def _attr(dataset: Dataset, name: str, default: object = np.nan) -> object:
    return _scalar(getattr(dataset, name, default), default)


def _float_attr(dataset: Dataset, name: str) -> float:
    try:
        return float(_attr(dataset, name))
    except (TypeError, ValueError):
        return float("nan")


def _variable(dataset: Dataset, *names: str) -> np.ndarray:
    lookup = {name.lower(): name for name in dataset.variables}
    for candidate in names:
        actual = lookup.get(candidate.lower())
        if actual is None:
            continue
        values = np.ma.filled(dataset.variables[actual][:], np.nan)
        return np.asarray(values, dtype=float).reshape(-1)
    return np.array([], dtype=float)


def _finite_stats(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    """计算数组有限值的数量、最小值、最大值和平均值。"""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_count": int(finite.size),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def _filename_fields(name: str, pattern: re.Pattern[str]) -> dict[str, str]:
    match = pattern.search(Path(name).name)
    return {} if match is None else {key: value for key, value in match.groupdict().items() if value}


def _gnss_id(dataset: Dataset, filename_fields: dict[str, str], ion: bool) -> str:
    if "gnss" in filename_fields:
        return filename_fields["gnss"].upper()
    prn_name = "occulting_sat_id" if ion else "prn_id"
    prn = int(_float_attr(dataset, prn_name))
    constellation = str(_attr(dataset, "conid", "G")).strip().upper() or "G"
    return f"{constellation[0]}{prn:02d}"


def _schema_text(item: NcItem, dataset: Dataset) -> str:
    lines = [f"file: {item.name}", f"archive: {item.archive}", "", "global attributes:"]
    for name in dataset.ncattrs():
        lines.append(f"  {name}: {getattr(dataset, name)!r}")
    lines.extend(["", "variables:"])
    for name, variable in dataset.variables.items():
        lines.append(
            f"  {name}: shape={variable.shape}, dtype={variable.dtype}, "
            f"units={getattr(variable, 'units', '')!r}"
        )
    return "\n".join(lines)


# 读取 podTc2 的元数据和主要变量，生成文件级摘要并保留少量绘图样本。
def summarize_pod(
    source: Path, max_files: int, sample_count: int
) -> tuple[pd.DataFrame, list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    schema = ""
    for index, item in enumerate(iter_nc_items(source, "podTc2")):
        if max_files and index >= max_files:
            break
        with open_item(item) as dataset:
            if not schema:
                schema = _schema_text(item, dataset)
            fields = _filename_fields(item.name, POD_NAME)
            tec = _variable(dataset, "TEC", "tec")
            elevation = _variable(dataset, "elevation", "elev")
            time = _variable(dataset, "time")
            s4_l1 = _variable(dataset, "S4_L1", "s4l1", "s4")
            start = _float_attr(dataset, "start_time")
            stop = _float_attr(dataset, "stop_time")
            row: dict[str, object] = {
                "archive": item.archive,
                "file": item.name,
                "leo": fields.get("leo", str(_attr(dataset, "leo_id", ""))),
                "gnss": _gnss_id(dataset, fields, ion=False),
                "antenna": fields.get("antenna", _attr(dataset, "antenna_id", "")),
                "start_gps_seconds": min(start, stop),
                "stop_gps_seconds": max(start, stop),
                "duration_seconds": abs(stop - start),
                "podflag": _attr(dataset, "podflag"),
                "attflag": _attr(dataset, "attflag"),
                "leodcb_flag": _attr(dataset, "leodcb_flag"),
                "gpsdcb_flag": _attr(dataset, "gpsdcb_flag"),
                "leveling_err_tecu": _float_attr(dataset, "leveling_err"),
            }
            row.update(_finite_stats(tec, "tec_tecu"))
            row.update(_finite_stats(elevation, "elevation_deg"))
            row.update(_finite_stats(s4_l1, "s4_l1"))
            rows.append(row)
            if len(samples) < sample_count:
                samples.append(
                    {"file": item.name, "time": time, "tec": tec, "elevation": elevation}
                )
    return pd.DataFrame(rows), samples, schema


def summarize_ion(source: Path, max_files: int) -> tuple[pd.DataFrame, str]:
    """汇总 ionPrf 的标识、时间范围和文件结构。"""

    rows: list[dict[str, object]] = []
    schema = ""
    for index, item in enumerate(iter_nc_items(source, "ionPrf")):
        if max_files and index >= max_files:
            break
        with open_item(item) as dataset:
            if not schema:
                schema = _schema_text(item, dataset)
            fields = _filename_fields(item.name, ION_NAME)
            bottom = _float_attr(dataset, "bottime")
            top = _float_attr(dataset, "toptime")
            rows.append(
                {
                    "archive": item.archive,
                    "file": item.name,
                    "leo": fields.get("leo", ""),
                    "gnss": _gnss_id(dataset, fields, ion=True),
                    "start_gps_seconds": min(bottom, top),
                    "stop_gps_seconds": max(bottom, top),
                    "edmax_gps_seconds": _float_attr(dataset, "edmaxtime"),
                    "setting": _attr(dataset, "setting"),
                    "icalib": _attr(dataset, "icalib"),
                    "tec0_tecu": _float_attr(dataset, "tec0"),
                    "tec1_tecu": _float_attr(dataset, "tec1"),
                    "bottom_alt_km": _float_attr(dataset, "botalt"),
                    "top_alt_km": _float_attr(dataset, "topalt"),
                    "edmax_alt_km": _float_attr(dataset, "edmaxalt"),
                    "edmax_lat_deg": _float_attr(dataset, "edmaxlat"),
                    "edmax_lon_deg": _float_attr(dataset, "edmaxlon"),
                }
            )
    return pd.DataFrame(rows), schema


# 按 LEO、GNSS 标识及 GPS 时间重叠关系建立 ionPrf 到 podTc2 的匹配表。
def match_ion_to_pod(
    ion: pd.DataFrame, pod: pd.DataFrame, margin_seconds: float
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    grouped = {
        key: group.reset_index(drop=True)
        for key, group in pod.groupby(["leo", "gnss"], dropna=False)
    }
    for ion_row in ion.itertuples(index=False):
        candidates = grouped.get((ion_row.leo, ion_row.gnss), pd.DataFrame())
        base: dict[str, object] = {
            "ion_file": ion_row.file,
            "leo": ion_row.leo,
            "gnss": ion_row.gnss,
            "ion_start_gps_seconds": ion_row.start_gps_seconds,
            "ion_stop_gps_seconds": ion_row.stop_gps_seconds,
            "ion_edmax_gps_seconds": ion_row.edmax_gps_seconds,
            "candidate_count_same_leo_gnss": len(candidates),
        }
        if candidates.empty:
            output.append({**base, "match_status": "no_same_leo_gnss"})
            continue
        left_gap = candidates["start_gps_seconds"] - ion_row.stop_gps_seconds
        right_gap = ion_row.start_gps_seconds - candidates["stop_gps_seconds"]
        gap = np.maximum(np.maximum(left_gap, right_gap), 0.0)
        overlap = np.minimum(candidates["stop_gps_seconds"], ion_row.stop_gps_seconds) - np.maximum(
            candidates["start_gps_seconds"], ion_row.start_gps_seconds
        )
        midpoint_distance = np.abs(
            (candidates["start_gps_seconds"] + candidates["stop_gps_seconds"]) / 2.0
            - ion_row.edmax_gps_seconds
        )
        eligible = gap <= margin_seconds
        if not eligible.any():
            nearest = int(gap.idxmin())
            output.append(
                {
                    **base,
                    "match_status": "no_time_match",
                    "nearest_gap_seconds": float(gap.loc[nearest]),
                }
            )
            continue
        ranking = pd.DataFrame(
            {"overlap": overlap, "distance": midpoint_distance}, index=candidates.index
        ).loc[eligible]
        best_index = int(
            ranking.sort_values(["overlap", "distance"], ascending=[False, True]).index[0]
        )
        best = candidates.loc[best_index]
        status = "overlap" if overlap.loc[best_index] >= 0.0 else "within_margin"
        output.append(
            {
                **base,
                "match_status": status,
                "pod_file": best["file"],
                "pod_archive": best["archive"],
                "pod_start_gps_seconds": best["start_gps_seconds"],
                "pod_stop_gps_seconds": best["stop_gps_seconds"],
                "time_gap_seconds": float(gap.loc[best_index]),
                "overlap_seconds": float(max(overlap.loc[best_index], 0.0)),
                "pod_tec_min_tecu": best["tec_tecu_min"],
                "pod_tec_max_tecu": best["tec_tecu_max"],
                "pod_elevation_min_deg": best["elevation_deg_min"],
                "pod_elevation_max_deg": best["elevation_deg_max"],
            }
        )
    return pd.DataFrame(output)


def plot_pod_examples(samples: list[dict[str, object]], output: Path, dpi: int) -> None:
    """绘制若干 POD 样本的 TEC 与仰角时间序列。"""

    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 1, figsize=(13, 3.2 * len(samples)), squeeze=False)
    for axis, sample in zip(axes[:, 0], samples):
        tec = np.asarray(sample["tec"], dtype=float)
        time = np.asarray(sample["time"], dtype=float)
        if time.size != tec.size:
            time = np.arange(tec.size, dtype=float)
            xlabel = "Sample index"
        else:
            time = (time - time[0]) / 60.0
            xlabel = "Minutes from arc start"
        axis.plot(time, tec, color="#2878b5", linewidth=1.1, label="TEC")
        axis.set(xlabel=xlabel, ylabel="TEC (TECU)")
        axis.grid(True, alpha=0.25)
        elevation = np.asarray(sample["elevation"], dtype=float)
        if elevation.size == tec.size:
            twin = axis.twinx()
            twin.plot(time, elevation, color="#d65f32", linewidth=0.9, alpha=0.75, label="Elevation")
            twin.set_ylabel("Elevation (deg)")
        axis.set_title(Path(str(sample["file"])).name, fontsize=9)
    fig.suptitle("COSMIC-2 podTc2 example tracking arcs")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary(pod: pd.DataFrame, matches: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制 POD 时长、TEC、最大仰角和匹配数量的汇总分布。"""

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    axes[0, 0].hist(pod["duration_seconds"].dropna() / 60.0, bins=35, color="#2878b5")
    axes[0, 0].set(xlabel="Arc duration (min)", ylabel="Files", title="POD arc duration")
    axes[0, 1].hist(pod["tec_tecu_mean"].dropna(), bins=35, color="#55a868")
    axes[0, 1].set(xlabel="Mean TEC (TECU)", ylabel="Files", title="Arc-mean slant TEC")
    axes[1, 0].hist(pod["elevation_deg_max"].dropna(), bins=35, color="#c44e52")
    axes[1, 0].set(xlabel="Maximum elevation (deg)", ylabel="Files", title="Geometry")
    if matches.empty:
        axes[1, 1].text(0.5, 0.5, "No ionPrf matching requested", ha="center", va="center")
        axes[1, 1].set_axis_off()
    else:
        counts = matches["match_status"].value_counts()
        axes[1, 1].bar(counts.index, counts.values, color="#8172b2")
        axes[1, 1].set(ylabel="ionPrf files", title="ionPrf → podTc2 matching")
        axes[1, 1].tick_params(axis="x", rotation=20)
    for axis in axes.flat:
        axis.grid(True, alpha=0.2)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """定义并解析检查程序的命令行参数。"""

    parser = argparse.ArgumentParser(description="检查 COSMIC-2 podTc2，并与 ionPrf 匹配")
    parser.add_argument("--pod", type=Path, required=True, help="podTc2 文件、目录或每日 tar.gz")
    parser.add_argument("--ion", type=Path, default=None, help="可选：ionPrf 文件、目录或每日 tar.gz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pod-files", type=int, default=0, help="0 表示读取全部")
    parser.add_argument("--max-ion-files", type=int, default=0, help="0 表示读取全部")
    parser.add_argument("--sample-plots", type=int, default=6)
    parser.add_argument("--match-margin-seconds", type=float, default=120.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    """执行 POD/ionPrf 汇总、匹配并输出表格、结构说明和图件。"""

    args = parse_args()
    if args.max_pod_files < 0 or args.max_ion_files < 0 or args.sample_plots < 0:
        raise SystemExit("max-pod-files、max-ion-files 和 sample-plots 不能为负数")
    if args.match_margin_seconds < 0:
        raise SystemExit("match-margin-seconds 不能为负数")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取 podTc2：{args.pod}")
    pod, samples, pod_schema = summarize_pod(args.pod, args.max_pod_files, args.sample_plots)
    if pod.empty:
        raise SystemExit("没有找到 podTc2 netCDF 文件")
    pod.to_csv(args.output_dir / "pod_file_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "pod_schema.txt").write_text(pod_schema, encoding="utf-8")

    ion = pd.DataFrame()
    matches = pd.DataFrame()
    if args.ion is not None:
        print(f"读取 ionPrf：{args.ion}")
        ion, ion_schema = summarize_ion(args.ion, args.max_ion_files)
        if ion.empty:
            print("警告：没有找到 ionPrf netCDF 文件")
        else:
            ion.to_csv(args.output_dir / "ion_file_summary.csv", index=False, encoding="utf-8-sig")
            (args.output_dir / "ion_schema.txt").write_text(ion_schema, encoding="utf-8")
            matches = match_ion_to_pod(ion, pod, args.match_margin_seconds)
            matches.to_csv(args.output_dir / "pod_ion_matches.csv", index=False, encoding="utf-8-sig")

    plot_pod_examples(samples, args.output_dir / "pod_tec_examples.png", args.dpi)
    plot_summary(pod, matches, args.output_dir / "pod_summary.png", args.dpi)
    print(f"podTc2 文件数：{len(pod):,}")
    if not matches.empty:
        print(f"ionPrf 文件数：{len(ion):,}")
        print(matches["match_status"].value_counts(dropna=False).to_string())
    print(f"结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
