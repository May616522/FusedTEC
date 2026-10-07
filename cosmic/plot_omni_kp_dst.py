"""根据 OMNI 小时数据绘制全年 Kp 和 Dst 双面板时间序列。

图形参照给定样式：上图为红色 Kp，下图为蓝色 Dst，共享年月横轴。
脚本默认先计算日平均再绘图，使全年尺度的曲线与参考图一样更加平滑；
不添加特定天数范围的阴影、竖线或文字。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"F:\FusedTec\Data\Other\omni2_2021.dat.csv")
DEFAULT_OUTPUT = Path(r"F:\FusedTec\Results\omni_plots\omni_kp_dst_2021.png")


def parse_args() -> argparse.Namespace:
    """定义输入文件、输出图像、年份、时间分辨率和图像分辨率。"""

    parser = argparse.ArgumentParser(description="绘制 OMNI Kp/Dst 全年时间序列")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--time-resolution",
        choices=("daily", "hourly"),
        default="daily",
        help="daily 表示日平均（默认，更平滑）；hourly 表示原始小时值",
    )
    return parser.parse_args()


def load_omni(path: Path, year: int) -> pd.DataFrame:
    """读取 OMNI CSV，构造 UTC 时间并处理 Kp/Dst 填充值。"""

    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    frame = pd.read_csv(path)
    required = {"Year", "Decimal_Day", "Hour", "Kp_Index", "Dst_Index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OMNI 文件缺少字段：{', '.join(missing)}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["Year"].eq(year)].copy()
    if frame.empty:
        raise ValueError(f"输入文件中没有 {year} 年数据")

    frame["Kp_Index"] = frame["Kp_Index"].replace(99, np.nan)
    frame["Dst_Index"] = frame["Dst_Index"].replace(99999, np.nan)
    kp_valid = frame["Kp_Index"].dropna()
    if not kp_valid.empty and kp_valid.quantile(0.95) > 9.0:
        frame["Kp_Index"] = frame["Kp_Index"] / 10.0
    frame.loc[~frame["Kp_Index"].between(0.0, 9.0), "Kp_Index"] = np.nan
    frame.loc[~frame["Dst_Index"].between(-2500.0, 1000.0), "Dst_Index"] = np.nan

    start = pd.to_datetime(frame["Year"].astype(int).astype(str) + "-01-01")
    frame["datetime"] = (
        start
        + pd.to_timedelta(frame["Decimal_Day"] - 1, unit="D")
        + pd.to_timedelta(frame["Hour"], unit="h")
    )
    frame = (
        frame[["datetime", "Kp_Index", "Dst_Index"]]
        .dropna(subset=["datetime"])
        .groupby("datetime", as_index=False, sort=True)
        .median(numeric_only=True)
    )
    if frame["datetime"].duplicated().any():
        raise RuntimeError("时间聚合后仍存在重复时刻")
    return frame


def aggregate_for_plot(frame: pd.DataFrame, time_resolution: str) -> pd.DataFrame:
    """按绘图分辨率整理数据；默认用日平均抑制小时级锯齿波动。"""

    if time_resolution == "hourly":
        return frame.copy()
    if time_resolution != "daily":
        raise ValueError(f"不支持的时间分辨率：{time_resolution}")

    # Kp 在 OMNI 中通常每 3 小时更新，直接绘制全年小时值会产生密集跳变。
    # 对 Kp 和 Dst 统一取日平均，既保留磁暴的日期变化，也更符合参考图的视觉尺度。
    daily = (
        frame.set_index("datetime")[["Kp_Index", "Dst_Index"]]
        .resample("1D")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    return daily


def style_axis(axis: plt.Axes) -> None:
    """设置与参考图一致的黑色边框、内向刻度和浅色网格。"""

    axis.tick_params(
        which="major",
        direction="in",
        length=5,
        width=0.9,
        top=True,
        right=True,
        labelsize=9,
    )
    axis.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.8,
        top=True,
        right=True,
    )
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    axis.grid(True, which="major", color="#D0D0D0", linewidth=0.65, alpha=0.85)


def plot_omni(frame: pd.DataFrame, year: int, output: Path, dpi: int) -> None:
    """绘制紧凑的 Kp/Dst 双面板全年时间序列并保存 PNG。"""

    if dpi < 50:
        raise ValueError("dpi 必须大于或等于 50")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 3.35),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )

    axes[0].plot(
        frame["datetime"],
        frame["Kp_Index"],
        color="red",
        linewidth=0.85,
        label="Kp",
    )
    axes[1].plot(
        frame["datetime"],
        frame["Dst_Index"],
        color="blue",
        linewidth=0.8,
        label="Dst",
    )

    axes[0].set_ylabel("Kp")
    # 根据当前年份的日平均 Kp 自动确定上限，不照搬参考图的固定范围。
    kp_max = float(frame["Kp_Index"].max())
    kp_upper = max(1.0, np.ceil((kp_max * 1.12) * 2.0) / 2.0)
    axes[0].set_ylim(0.0, kp_upper)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].legend(loc="upper left", frameon=False, handlelength=2.4)

    axes[1].set_ylabel("Dst (nT)")
    # Dst 上下各保留约 10% 空间，再取整到 10 nT，避免大片无数据区域。
    dst_min = float(frame["Dst_Index"].min())
    dst_max = float(frame["Dst_Index"].max())
    dst_span = max(dst_max - dst_min, 10.0)
    dst_lower = 10.0 * np.floor((dst_min - 0.10 * dst_span) / 10.0)
    dst_upper = 10.0 * np.ceil((dst_max + 0.10 * dst_span) / 10.0)
    axes[1].set_ylim(dst_lower, dst_upper)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[1].legend(loc="upper left", frameon=False, handlelength=2.4)
    axes[1].set_xlabel("Time (Year-Month)")

    for axis in axes:
        style_axis(axis)
        # 两端各留少量空白，防止起止月份标签在保存图像时被裁切。
        axis.set_xlim(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10))
        axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
        axis.xaxis.set_minor_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(labelbottom=False)
    figure.align_ylabels(axes)
    figure.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.19)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> None:
    """读取数据、打印核对信息并生成图像。"""

    args = parse_args()
    hourly_frame = load_omni(args.input, args.year)
    plot_frame = aggregate_for_plot(hourly_frame, args.time_resolution)
    plot_omni(plot_frame, args.year, args.output, args.dpi)
    resolution_text = "日平均" if args.time_resolution == "daily" else "小时值"
    print(
        f"已绘制 {len(plot_frame):,} 个{resolution_text}点："
        f"Kp=[{plot_frame['Kp_Index'].min():.2f}, {plot_frame['Kp_Index'].max():.2f}]，"
        f"Dst=[{plot_frame['Dst_Index'].min():.2f}, {plot_frame['Dst_Index'].max():.2f}] nT",
        flush=True,
    )
    print(f"图像已保存：{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
