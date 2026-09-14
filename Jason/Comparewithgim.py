"""Interpolate GIM VTEC for Jason observations and create residual diagnostics.

Outputs: enriched CSV files, one daily-statistics CSV, one daily line chart,
and one observation-level global residual map. Residual = GIM VTEC - Jason VTEC.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from helper.interpolator import GIMInterpolator


REQUIRED_COLUMNS = ("datetime", "lat", "lon")
DAILY_STATS_NAME = "daily_residual_statistics.csv"


#检查是否缺失必要列
def _validate_columns(df: pd.DataFrame, compare_col: str, csv_path: Path) -> None:
    required = {*REQUIRED_COLUMNS, compare_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name} 缺少列: {', '.join(missing)}")

#将csv中的经度统一到gim对应的 -180到180之间
def _normalise_longitude(longitude: pd.Series) -> pd.Series:
    """Convert either 0–360° or -180–180° longitude to -180–180°."""

    return (longitude.astype(float) + 180.0) % 360.0 - 180.0


#计算日均的误差
def _daily_statistics(accumulator: dict) -> pd.DataFrame:
    rows = []
    for date, (residual_sum, residual_square_sum, count) in accumulator.items():
        mean = residual_sum / count
        variance = max(residual_square_sum / count - mean**2, 0.0)
        rows.append(
            {
                "date": date,
                "mean": mean,
                "std": np.sqrt(variance),
                "rmse": np.sqrt(residual_square_sum / count),
                "count": int(count),
            }
        )

    columns = ["date", "mean", "std", "rmse", "count"]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("date")
        .reset_index(drop=True))


#进行插值计算，对文件夹中的每个文件都这么处理
def process_jason_folder(
    folder_path: str | Path,
    gim: "GIMInterpolator",
    compare_col: str = "TEC_smooth",
    output_folder: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    source_dir = Path(folder_path)
    destination_dir = (Path(output_folder) if output_folder is not None else source_dir / "compare_with_gim")
    destination_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{source_dir} 中没有 CSV 文件")

    daily_accumulator = defaultdict(lambda: np.zeros(3, dtype=float))
    point_frames: list[pd.DataFrame] = []
    failed_files: list[tuple[str, str]] = []
    valid_point_count = 0

    print(f"开始处理 {len(csv_files)} 个 Jason CSV")
    for index, csv_path in enumerate(csv_files, start=1):
        try:
            frame = pd.read_csv(csv_path)
            _validate_columns(frame, compare_col, csv_path)
            frame = gim.add_gim_to_dataframe(
                frame,
                time_col="datetime",
                lat_col="lat",
                lon_col="lon",
                output_col="gim_vtec",
            )
            
            #请注意，这里是什么减去什么
            frame["residual"] = frame["gim_vtec"] - frame[compare_col]
            #控制以下输出的东西，以后可以改的
            frame=frame[["datetime","lat","lon","TEC_raw","TEC_smooth","alt","gim_vtec","residual"]]
            frame.to_csv(destination_dir / csv_path.name, index=False)
            
            #从这里开始是对数据的统计，前面已经完成插值和输出
            valid = frame[["datetime", "lat", "lon", "residual"]].copy()
            valid["lon"] = _normalise_longitude(valid["lon"])
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
            valid = valid[
                valid["lat"].between(-90.0, 90.0)
                & valid["lon"].between(-180.0, 180.0)]
            if valid.empty:
                print(f"[{index}/{len(csv_files)}] {csv_path.name}: 无有效残差")
                continue
            
            #对日均值进行统计
            valid["date"] = pd.to_datetime(valid["datetime"]).dt.floor("D")#取到日期，抛弃小时
            for date, group in valid.groupby("date", sort=False):
                residual = group["residual"].to_numpy(dtype=float)
                daily_accumulator[date] += (
                    residual.sum(),
                    np.dot(residual, residual),
                    residual.size,
                )
                
            #这里是对所有数据进行汇合
            point_frames.append(valid[["lat", "lon", "residual"]])
            valid_point_count += len(valid)
            print(f"[{index}/{len(csv_files)}] {csv_path.name}: {len(valid)} 个有效点")
        except Exception as exc:  # One bad orbit should not discard the full batch.
            failed_files.append((csv_path.name, str(exc)))
            print(f"[{index}/{len(csv_files)}] {csv_path.name}: 失败 ({exc})")

    if not point_frames:
        details = "; ".join(f"{name}: {reason}" for name, reason in failed_files[:3])
        raise RuntimeError(f"没有得到有效残差数据。{details}")

    residual_points = pd.concat(point_frames, ignore_index=True)
    daily_stats = _daily_statistics(daily_accumulator)#每日有效数据
    statistics_dir = destination_dir / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    daily_stats.to_csv(statistics_dir / DAILY_STATS_NAME, index=False)

    print(
        f"处理完成：成功 {len(csv_files) - len(failed_files)} 个，"
        f"失败 {len(failed_files)} 个，有效点 {valid_point_count} 个")
    if failed_files:
        print("失败文件：")
        for name, reason in failed_files:
            print(f"  - {name}: {reason}")

    return residual_points, daily_stats


#保存以及展示选项
def _save_or_show(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


#利用每日统计的结果，绘制图象，注意不是原始数据绘制的图象
def plot_daily_analysis(
    daily_df: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = False,) -> None:
    """"绘制每日平均残差、±1 倍标准差和 RMSE 的综合对比图"""
    
    if daily_df.empty:
        raise ValueError("daily_df 为空")
    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    
    #绘制半透明蓝色标准差带
    # ax.fill_between(
    #     frame["date"],
    #     frame["mean"] - frame["std"],
    #     frame["mean"] + frame["std"],
    #     color="#79a9dc",
    #     alpha=0.24,
    #     linewidth=0,
    #     label="Mean ± 1 STD",)
    
    ax.plot(
        frame["date"],
        frame["mean"],
        color="#1f5a99",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Daily mean residual",)
    
    ax.plot(
        frame["date"],
        frame["rmse"],
        color="#d55e00",
        linewidth=1.6,
        linestyle="--",
        label="Daily RMSE",)
    
    ax.plot(
        frame["date"],
        frame["std"],
        color="#79a9dc",
        linewidth=1.6,
        linestyle="-.",
        label="Daily STD",)
    
    #在y=0.0处加了一条浅灰色虚线，判断残差大于或小于0
    ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
    ax.set(title="Daily GIM − Jason VTEC comparison", xlabel="Date", ylabel="TECU")#设置标题
    ax.grid(axis="y", color="0.85", linewidth=0.7)  # 只添加水平浅灰色网格，便于读数
    ax.spines[["top", "right"]].set_visible(False) #隐藏顶部和右侧边框（简洁风格
    ax.legend(frameon=False, ncols=3, loc="upper center") # 无边框图例，分3列显示在顶部中央
    
    #智能设置日期刻度，但是目前比较鸡肋，待调整
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    _save_or_show(fig, save_path, show)


#绘制全球分布的残差，这部分做的不好，待进步
def plot_global_residual(
    residual_points: pd.DataFrame,
    save_path: str | Path | None = None,
    color_limit: float | None = None,
    max_points: int = 1_000_000,
    show: bool = False,
) -> None:
    """Plot point-level residuals with coastlines, without rectangular binning."""

    if residual_points.empty:
        raise ValueError("residual_points 为空")

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    except ImportError as exc:
        raise ImportError("全球地图绘制需要安装 cartopy") from exc

    points = residual_points[["lon", "lat", "residual"]].dropna()
    if len(points) > max_points:
        points = points.sample(max_points, random_state=42)

    if color_limit is None:
        color_limit = float(np.nanpercentile(np.abs(points["residual"]), 98))
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 6.3), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#f8fbfd", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f4f1ea", zorder=0)
    scatter = ax.scatter(
        points["lon"],
        points["lat"],
        c=points["residual"],
        s=1.2,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
        transform=projection,
        linewidths=0,
        alpha=0.72,
        rasterized=True,
        zorder=2,
    )
    ax.coastlines(resolution="50m", color="0.08", linewidth=0.65, zorder=3)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        edgecolor="0.25",
        linewidth=0.35,
        zorder=3,
    )
    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=True,
        xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-60, 61, 30),
        linewidth=0.45,
        color="0.55",
        alpha=0.5,
        linestyle=":",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xformatter = LONGITUDE_FORMATTER
    gridlines.yformatter = LATITUDE_FORMATTER
    gridlines.xlabel_style = {"size": 8}
    gridlines.ylabel_style = {"size": 8}

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        orientation="vertical",
        shrink=0.84,
        pad=0.025,
    )
    colorbar.set_label("Residual: GIM − Jason (TECU)")
    ax.set_title("Global distribution of GIM − Jason VTEC residuals", pad=10)
    _save_or_show(fig, save_path, show)


def run_analysis(
    source_dir: str | Path,
    gim_dir: str | Path,
    output_dir: str | Path,
    compare_col: str = "TEC_smooth",
    color_limit: float | None = None,
    show: bool = False,
) -> None:
    """Run interpolation, CSV export, and the two requested analyses."""

    from helper.interpolator import GIMInterpolator

    output_dir = Path(output_dir)
    points, daily = process_jason_folder(
        folder_path=source_dir,
        gim=GIMInterpolator(gim_dir),
        compare_col=compare_col,
        output_folder=output_dir,
    )
    figure_dir = output_dir / "figures"
    plot_daily_analysis(daily, figure_dir / "daily_analysis.png", show=show)
    plot_global_residual(
        points,
        figure_dir / "global_residual_distribution.png",
        color_limit=color_limit,
        show=show,)
    print(f"插值 CSV：{output_dir}")
    print(f"逐日统计：{output_dir / 'statistics' / DAILY_STATS_NAME}")
    print(f"分析图：{figure_dir}")


#设置输入输出路径等默认参数
def _parse_arguments() -> argparse.Namespace:
    data_root = PROJECT_ROOT.parent / "Data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=data_root / "Jason2021" / "TecCSV2",
        help="Jason CSV 文件夹",
    )
    parser.add_argument(
        "--gim",
        type=Path,
        default=data_root / "GIM2021" / "IGS",
        help="GIM IONEX 根目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root / "Jason2021" / "TecCSV2_redisual1",
        help="输出文件夹",
    )
    parser.add_argument("--compare-col", default="TEC_smooth", help="Jason VTEC 列名")
    parser.add_argument(
        "--color-limit",
        type=float,
        default=None,
        help="地图色标的对称上下限；默认使用残差绝对值的第 98 百分位",
    )
    parser.add_argument("--show", action="store_true", help="保存后同时显示图片")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    run_analysis(
        source_dir=args.source.resolve(),
        gim_dir=args.gim.resolve(),
        output_dir=args.output.resolve(),
        compare_col=args.compare_col,
        color_limit=args.color_limit,
        show=args.show,
    )
