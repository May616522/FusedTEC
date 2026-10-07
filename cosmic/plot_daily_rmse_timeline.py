"""绘制 COSMIC XGBoost 验证集和测试集的逐日 RMSE 诊断图。

脚本直接读取训练结果中的 ``all_split_predictions.csv`` 和
``doy_split_assignments.csv``，重新计算每个完整 DOY 的 RMSE、MAE、Bias 和
R²。横轴使用真实日期，没有样本的日期保持为空，避免把不连续日期错误连接。

如果提供 OMNI 文件，脚本还会按“当日最大 Kp 或当日最小 Dst”识别磁暴日，
并在时间图和分组分布图中单独标记。磁暴阈值可通过命令行修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


DEFAULT_RESULT_DIR = Path(r"F:\FusedTec\Results\xg_cosmic_results\2836505")
DEFAULT_OMNI_FILE = Path(r"F:\FusedTec\Data\Other\omni2_2021.dat.csv")
REQUIRED_PREDICTION_COLUMNS = {
    "split",
    "source_doy",
    "true_residual",
    "predicted_residual",
    "prediction_error",
}


def parse_args() -> argparse.Namespace:
    """定义输入路径、磁暴阈值和绘图参数。"""

    parser = argparse.ArgumentParser(
        description="绘制完整 DOY 验证/测试 RMSE 时间分布和分组分布"
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--omni-file", type=Path, default=DEFAULT_OMNI_FILE)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--kp-storm-threshold", type=float, default=5.0)
    parser.add_argument("--dst-storm-threshold", type=float, default=-50.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """检查结果文件并确定输出目录。"""

    prediction_path = args.result_dir / "all_split_predictions.csv"
    assignment_path = args.result_dir / "doy_split_assignments.csv"
    for path in (prediction_path, assignment_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    if args.dpi < 50:
        raise ValueError("dpi 必须大于或等于 50")
    output_dir = args.output_dir or args.result_dir / "daily_rmse_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return prediction_path, assignment_path, output_dir


def safe_daily_r2(group: pd.DataFrame) -> float:
    """在目标存在变化时计算单日 R²，否则返回 NaN。"""

    true = group["true_residual"].to_numpy(dtype=float)
    predicted = group["predicted_residual"].to_numpy(dtype=float)
    if len(group) < 2 or np.std(true) == 0.0:
        return float("nan")
    return float(r2_score(true, predicted))


def calculate_daily_metrics(predictions: pd.DataFrame, year: int) -> pd.DataFrame:
    """根据逐记录预测重新计算每个 split、每个 DOY 的评价指标。"""

    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"预测文件缺少字段：{', '.join(missing)}")
    predictions = predictions.copy()
    for column in (
        "source_doy",
        "true_residual",
        "predicted_residual",
        "prediction_error",
    ):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    predictions = predictions.dropna(subset=list(REQUIRED_PREDICTION_COLUMNS))
    predictions["squared_error"] = predictions["prediction_error"] ** 2
    grouped = predictions.groupby(["split", "source_doy"], sort=True)
    daily = grouped.agg(
        rows=("prediction_error", "size"),
        true_mean=("true_residual", "mean"),
        predicted_mean=("predicted_residual", "mean"),
        bias=("prediction_error", "mean"),
        mae=("prediction_error", lambda values: float(np.mean(np.abs(values)))),
        mean_squared_error=("squared_error", "mean"),
    ).reset_index()
    daily["rmse"] = np.sqrt(daily.pop("mean_squared_error"))
    r2_values = grouped.apply(safe_daily_r2, include_groups=False).rename("r2")
    daily = daily.merge(
        r2_values.reset_index(), on=["split", "source_doy"], validate="one_to_one"
    )
    daily["source_doy"] = daily["source_doy"].astype(int)
    daily["date"] = pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(
        daily["source_doy"] - 1, unit="D"
    )
    return daily


def add_fold_assignments(daily: pd.DataFrame, assignment_path: Path) -> pd.DataFrame:
    """把每个开发日所属的 GroupKFold 验证折加入逐日统计。"""

    assignments = pd.read_csv(assignment_path)
    if "doy" not in assignments.columns:
        raise ValueError("DOY 划分表缺少 doy 字段")
    assignments["doy"] = pd.to_numeric(assignments["doy"], errors="coerce")
    selected = ["doy"]
    for column in ("cv_validation_fold", "outer_split"):
        if column in assignments.columns:
            selected.append(column)
    return daily.merge(
        assignments[selected],
        left_on="source_doy",
        right_on="doy",
        how="left",
        validate="many_to_one",
    ).drop(columns="doy")


def load_omni_daily(
    path: Path,
    year: int,
    kp_threshold: float,
    dst_threshold: float,
) -> pd.DataFrame:
    """从 OMNI 小时数据计算逐日 Kp 最大值、Dst 最小值和磁暴标志。"""

    if not path.is_file():
        print(f"警告：未找到 OMNI 文件，不绘制磁暴标记：{path}", flush=True)
        return pd.DataFrame(columns=["source_doy", "kp_max", "dst_min", "is_storm"])
    omni = pd.read_csv(path)
    required = {"Year", "Decimal_Day", "Kp_Index", "Dst_Index"}
    missing = sorted(required.difference(omni.columns))
    if missing:
        raise ValueError(f"OMNI 文件缺少字段：{', '.join(missing)}")
    for column in required:
        omni[column] = pd.to_numeric(omni[column], errors="coerce")
    omni = omni.loc[omni["Year"].eq(year)].copy()
    daily = (
        omni.groupby("Decimal_Day", as_index=False)
        .agg(kp_max=("Kp_Index", "max"), dst_min=("Dst_Index", "min"))
        .rename(columns={"Decimal_Day": "source_doy"})
    )
    daily["source_doy"] = daily["source_doy"].astype(int)
    daily["is_storm"] = (daily["kp_max"] >= kp_threshold) | (
        daily["dst_min"] <= dst_threshold
    )
    return daily


def calculate_overall_rmse(predictions: pd.DataFrame) -> dict[str, float]:
    """计算每个 split 基于全部记录的总体 RMSE，用作水平参考线。"""

    return {
        str(split): float(np.sqrt(np.mean(group["prediction_error"] ** 2)))
        for split, group in predictions.groupby("split")
    }


def plot_timeline(
    daily: pd.DataFrame,
    overall_rmse: dict[str, float],
    year: int,
    output: Path,
    dpi: int,
) -> None:
    """绘制验证 OOF 和测试集共享真实日期轴的逐日 RMSE 散点图。"""

    development = daily.loc[daily["split"].eq("development_oof")].copy()
    test = daily.loc[daily["split"].eq("test")].copy()
    if development.empty or test.empty:
        raise ValueError("预测文件必须同时包含 development_oof 和 test")

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    fold_colors = {1: "#2878B5", 2: "#F08A24", 3: "#3A923A"}
    if "cv_validation_fold" in development.columns:
        for fold, group in development.groupby("cv_validation_fold", dropna=True):
            fold_number = int(fold)
            axes[0].scatter(
                group["date"],
                group["rmse"],
                s=18,
                alpha=0.72,
                color=fold_colors.get(fold_number, "#666666"),
                label=f"CV fold {fold_number}",
            )
    else:
        axes[0].scatter(
            development["date"], development["rmse"], s=18, alpha=0.72
        )
    axes[1].scatter(
        test["date"],
        test["rmse"],
        s=45,
        marker="^",
        color="#C83E4D",
        alpha=0.9,
        label="Independent test DOY",
    )

    # 磁暴日使用空心星号覆盖，不改变原散点所表达的数据集合。
    if "is_storm" in daily.columns:
        for axis, frame in zip(axes, (development, test)):
            storm = frame.loc[frame["is_storm"].fillna(False)]
            if not storm.empty:
                axis.scatter(
                    storm["date"],
                    storm["rmse"],
                    s=115,
                    marker="*",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    label="Storm DOY",
                    zorder=5,
                )

    for axis, split, label in (
        (axes[0], "development_oof", "Development OOF"),
        (axes[1], "test", "Independent test"),
    ):
        if split in overall_rmse:
            axis.axhline(
                overall_rmse[split],
                color="black",
                linestyle="--",
                linewidth=1.1,
                label=f"Overall RMSE = {overall_rmse[split]:.3f}",
            )
        axis.set_ylabel("Daily RMSE (TECU)")
        axis.set_title(label)
        axis.grid(alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="upper left", ncol=2)

    axes[1].set_xlim(
        pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=3),
        pd.Timestamp(year=year, month=12, day=31) + pd.Timedelta(days=3),
    )
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].set_xlabel(f"Date in {year}; missing dates are intentionally left blank")
    figure.suptitle("Daily residual-prediction RMSE on complete DOYs", fontsize=16)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_grouped_distribution(daily: pd.DataFrame, output: Path, dpi: int) -> None:
    """用箱线图和抖动散点比较验证/测试及平静/磁暴日期。"""

    if "is_storm" not in daily.columns or daily["is_storm"].isna().all():
        groups = [
            ("OOF", daily.loc[daily["split"].eq("development_oof"), "rmse"]),
            ("Test", daily.loc[daily["split"].eq("test"), "rmse"]),
        ]
    else:
        storm = daily["is_storm"].fillna(False)
        groups = [
            (
                "OOF quiet",
                daily.loc[daily["split"].eq("development_oof") & ~storm, "rmse"],
            ),
            (
                "OOF storm",
                daily.loc[daily["split"].eq("development_oof") & storm, "rmse"],
            ),
            ("Test quiet", daily.loc[daily["split"].eq("test") & ~storm, "rmse"]),
            ("Test storm", daily.loc[daily["split"].eq("test") & storm, "rmse"]),
        ]
    groups = [(label, values.dropna().to_numpy()) for label, values in groups if len(values)]
    labels = [label for label, _ in groups]
    values = [value for _, value in groups]
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    box = axis.boxplot(values, showfliers=False, patch_artist=True)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    colors = ["#77AADD", "#EE8866", "#88CCAA", "#CC6677"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(42)
    for position, value in enumerate(values, start=1):
        jitter = rng.normal(position, 0.055, size=len(value))
        axis.scatter(jitter, value, s=15, alpha=0.48, color="#333333")
        axis.text(
            position,
            axis.get_ylim()[1],
            f"n={len(value)}",
            ha="center",
            va="top",
            fontsize=9,
        )
    axis.set(
        ylabel="Daily RMSE (TECU)",
        title="Distribution of daily RMSE by evaluation subset",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main() -> None:
    """读取数据、生成逐日统计并输出两类诊断图。"""

    args = parse_args()
    prediction_path, assignment_path, output_dir = validate_inputs(args)
    predictions = pd.read_csv(prediction_path)
    daily = calculate_daily_metrics(predictions, args.year)
    daily = add_fold_assignments(daily, assignment_path)
    omni_daily = load_omni_daily(
        args.omni_file,
        args.year,
        args.kp_storm_threshold,
        args.dst_storm_threshold,
    )
    daily = daily.merge(omni_daily, on="source_doy", how="left", validate="many_to_one")
    overall_rmse = calculate_overall_rmse(predictions)

    daily.to_csv(
        output_dir / "daily_rmse_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    summary = (
        daily.assign(
            activity=np.where(daily["is_storm"].fillna(False), "storm", "quiet")
        )
        .groupby(["split", "activity"], as_index=False)
        .agg(
            days=("source_doy", "nunique"),
            mean_daily_rmse=("rmse", "mean"),
            median_daily_rmse=("rmse", "median"),
            min_daily_rmse=("rmse", "min"),
            max_daily_rmse=("rmse", "max"),
        )
    )
    summary.to_csv(
        output_dir / "daily_rmse_summary.csv", index=False, encoding="utf-8-sig"
    )
    plot_timeline(
        daily,
        overall_rmse,
        args.year,
        output_dir / "daily_rmse_timeline.png",
        args.dpi,
    )
    plot_grouped_distribution(
        daily, output_dir / "daily_rmse_distribution.png", args.dpi
    )
    print(f"逐日 RMSE 图和统计表已保存：{output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
