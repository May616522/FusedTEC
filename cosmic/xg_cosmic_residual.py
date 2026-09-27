"""训练 COSMIC-2 residual 的 XGBoost 回归模型。

目标值为 ``residual``。数据按完整年积日随机分为训练、验证、测试集，任何
年积日都只属于一个集合。模型最多训练 500 轮，并通过按完整 DOY 进行的
3 折 GroupKFold、验证折早停、L1、L2、行采样和列采样抑制过拟合。为减轻
预测向均值收缩的问题，仅根据每一折训练集目标分布识别两端 residual 样本并
提高其训练权重。最终测试集始终独立，不参与交叉验证、权重阈值或轮次选择。

Kp 和 Dst 的滞后特征在运行时直接从逐小时 OMNI 文件生成，避免修改已经生成
的 COSMIC 日级 CSV。所有滞后均以观测 UTC 时刻为基准，正值表示使用过去的
空间天气状态，绝不使用未来数据。

注意：``residual`` 只作为目标值，绝不能同时作为输入特征，否则会产生直接
目标泄漏。树模型不需要标准化，因此本脚本直接使用已经生成的周期特征。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


TARGET_COLUMN = "residual"
DEFAULT_INPUT_DIR = Path(
    "/share/home/u23114/tj23114/data/Yaoyaping_data/cosmic2021"
)
DEFAULT_OUTPUT_DIR = Path(
    "/share/home/u23114/tj23114/packages/yaoyaping/cosmic2021/xg_cosmic_results"
)
DEFAULT_OMNI_FILE = Path(
    "/share/home/u23114/tj23114/data/Yaoyaping_data/Other/omni2_2021.dat.csv"
)
BASE_FEATURE_COLUMNS = [
    "mag_lon_c",
    "mag_lon_s",
    "mag_lat_s",
    "mag_lat_c",
    "DOY_s",
    "DOY_c",
    "local_time_s",
    "local_time_c",
    "tec0",
    "Kp_Index",
    "Dst_Index",
    "f10.7_Index",
    "is_land",
]
LAG_FEATURE_SPECS = {
    "Kp_3h_lag": ("Kp_Index", 3),
    "Kp_6h_lag": ("Kp_Index", 6),
    "Kp_12h_lag": ("Kp_Index", 12),
    "Dst_1h_lag": ("Dst_Index", 1),
    "Dst_3h_lag": ("Dst_Index", 3),
    "Dst_6h_lag": ("Dst_Index", 6),
}
FEATURE_COLUMNS = [*BASE_FEATURE_COLUMNS, *LAG_FEATURE_SPECS]
SPLIT_ORDER = ("train", "validation", "test")
SCIENCE_PATTERN = re.compile(r"^cosmic2_model_(\d{4})_(\d{3})\.csv$")


@dataclass(frozen=True)
class SplitMetrics:
    """保存一个数据集合上的回归评价指标。"""

    rows: int
    days: int
    rmse: float
    mae: float
    r2: float
    bias: float
    correlation: float


@dataclass(frozen=True)
class OmniLagSource:
    """保存生成滞后特征所需的 OMNI 小时时间轴。"""

    timestamps: np.ndarray
    values: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    """定义训练、分组、正则化和输出参数。"""

    parser = argparse.ArgumentParser(
        description="按完整年积日分组训练 COSMIC residual XGBoost 模型"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--omni-file", type=Path, default=DEFAULT_OMNI_FILE)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="在非测试 DOY 上进行的 GroupKFold 折数",
    )
    parser.add_argument(
        "--max-rows-per-day",
        type=int,
        default=0,
        help="每个年积日最多随机保留的行数；0 表示保留全部",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.10,
        help="训练集两端加权所用分位数；0.10 表示最低和最高各 10%%",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=2.0,
        help="训练集两端 residual 样本权重；中间样本权重固定为 1",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--reg-alpha", type=float, default=0.10, help="L1 正则化")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 正则化")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-plot-points", type=int, default=200_000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在读取大数据前检查路径和参数范围。"""

    if not args.input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{args.input_dir}")
    if not args.omni_file.is_file():
        raise ValueError(f"OMNI 文件不存在：{args.omni_file}")
    ratios = np.array(
        [args.train_ratio, args.validation_ratio, args.test_ratio], dtype=float
    )
    if not np.isfinite(ratios).all() or np.any(ratios <= 0.0):
        raise ValueError("训练/验证/测试比例必须为正数")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("训练/验证/测试比例之和必须为 1")
    if args.cv_folds < 2:
        raise ValueError("cv-folds 必须大于或等于 2")
    if args.n_estimators < 1:
        raise ValueError("n-estimators 必须大于 0")
    if args.early_stopping_rounds < 0:
        raise ValueError("early-stopping-rounds 不能为负")
    if not 0.0 < args.tail_quantile < 0.5:
        raise ValueError("tail-quantile 必须位于 (0, 0.5)")
    if not np.isfinite(args.tail_weight) or args.tail_weight < 1.0:
        raise ValueError("tail-weight 必须是大于或等于 1 的有限数")
    if args.reg_alpha <= 0.0 or args.reg_lambda <= 0.0:
        raise ValueError("reg-alpha 和 reg-lambda 必须都大于 0，以启用 L1/L2")
    if not 0.0 < args.learning_rate <= 1.0:
        raise ValueError("learning-rate 必须位于 (0, 1]")
    if not 0.0 < args.subsample <= 1.0:
        raise ValueError("subsample 必须位于 (0, 1]")
    if not 0.0 < args.colsample_bytree <= 1.0:
        raise ValueError("colsample-bytree 必须位于 (0, 1]")
    if args.max_rows_per_day < 0:
        raise ValueError("max-rows-per-day 不能为负")


def discover_files(input_dir: Path, year: int) -> list[tuple[int, Path]]:
    """发现严格匹配指定年份的日级科学 CSV。"""

    selected: list[tuple[int, Path]] = []
    for path in input_dir.glob(f"cosmic2_model_{year}_???.csv"):
        match = SCIENCE_PATTERN.match(path.name)
        if match and int(match.group(1)) == year:
            selected.append((int(match.group(2)), path))
    selected.sort()
    if len(selected) < 3:
        raise ValueError("至少需要 3 个年积日才能建立训练/验证/测试集")
    duplicated = pd.Series([doy for doy, _ in selected]).duplicated()
    if duplicated.any():
        raise ValueError("发现重复年积日文件")
    return selected


def load_omni_lag_source(path: Path, year: int) -> OmniLagSource:
    """读取并校验逐小时 OMNI Kp/Dst，为滞后查询建立有序时间轴。"""

    frame = pd.read_csv(path)
    required = {"Year", "Decimal_Day", "Hour", "Kp_Index", "Dst_Index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OMNI 文件缺少字段：{', '.join(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["Year"].eq(year)].copy()
    frame = frame.dropna(subset=["Year", "Decimal_Day", "Hour"])
    if frame.empty:
        raise ValueError(f"OMNI 文件中没有 {year} 年数据")

    base_dates = pd.to_datetime(
        frame["Year"].astype(int).astype(str) + "-01-01", utc=True, errors="coerce"
    )
    frame["datetime"] = (
        base_dates
        + pd.to_timedelta(frame["Decimal_Day"] - 1.0, unit="D")
        + pd.to_timedelta(frame["Hour"], unit="h")
    )
    frame["Kp_Index"] = frame["Kp_Index"].replace(99, np.nan)
    frame["Dst_Index"] = frame["Dst_Index"].replace(99999, np.nan)
    kp_valid = frame["Kp_Index"].dropna()
    if not kp_valid.empty and kp_valid.quantile(0.95) > 9.0:
        frame["Kp_Index"] = frame["Kp_Index"] / 10.0
    frame.loc[~frame["Kp_Index"].between(0.0, 9.0), "Kp_Index"] = np.nan
    frame.loc[~frame["Dst_Index"].between(-2500.0, 1000.0), "Dst_Index"] = np.nan
    frame = (
        frame[["datetime", "Kp_Index", "Dst_Index"]]
        .dropna(subset=["datetime"])
        .groupby("datetime", as_index=False, sort=True)
        .median(numeric_only=True)
    )
    timestamps = frame["datetime"].astype("int64").to_numpy(dtype=np.float64) / 1e9
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("OMNI 时间轴为空、重复或未严格递增")
    return OmniLagSource(
        timestamps=timestamps,
        values={
            column: frame[column].to_numpy(dtype=float)
            for column in ("Kp_Index", "Dst_Index")
        },
    )


def interpolate_omni_lag(
    observation_seconds: np.ndarray,
    source_seconds: np.ndarray,
    source_values: np.ndarray,
    lag_hours: int,
    max_gap_hours: float = 3.0,
) -> np.ndarray:
    """在过去的指定时刻插值 OMNI，并拒绝越界或跨越过大缺口的查询。"""

    query = np.asarray(observation_seconds, dtype=np.float64) - lag_hours * 3600.0
    valid_source = np.isfinite(source_seconds) & np.isfinite(source_values)
    x = source_seconds[valid_source]
    y = source_values[valid_source]
    result = np.full(query.shape, np.nan, dtype=float)
    valid_query = np.isfinite(query)
    if x.size == 0 or not valid_query.any():
        return result

    query_indices = np.flatnonzero(valid_query)
    query_values = query[query_indices]
    positions = np.searchsorted(x, query_values, side="left")
    clipped = np.minimum(positions, x.size - 1)
    exact = (positions < x.size) & (x[clipped] == query_values)
    result[query_indices[exact]] = y[clipped[exact]]

    between = (~exact) & (positions > 0) & (positions < x.size)
    selected_local = np.flatnonzero(between)
    if selected_local.size:
        right = positions[selected_local]
        left = right - 1
        gaps = x[right] - x[left]
        allowed = gaps <= max_gap_hours * 3600.0
        selected_local = selected_local[allowed]
        left = left[allowed]
        right = right[allowed]
        fraction = (query_values[selected_local] - x[left]) / (x[right] - x[left])
        result[query_indices[selected_local]] = y[left] + fraction * (y[right] - y[left])
    return result


def load_data(
    files: list[tuple[int, Path]],
    omni: OmniLagSource,
    max_rows_per_day: int,
    sampling_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取特征和目标，生成 OMNI 滞后量，并可在每个 DOY 内抽样。"""

    source_required = [*BASE_FEATURE_COLUMNS, TARGET_COLUMN, "datetime"]
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    for position, (doy, path) in enumerate(files, start=1):
        frame = pd.read_csv(path, usecols=lambda column: column in source_required)
        missing = sorted(set(source_required).difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} 缺少字段：{', '.join(missing)}")
        original_rows = len(frame)
        observation_time = pd.to_datetime(frame.pop("datetime"), utc=True, errors="coerce")
        observation_seconds = np.full(len(frame), np.nan, dtype=np.float64)
        valid_time = observation_time.notna().to_numpy()
        observation_seconds[valid_time] = (
            observation_time.loc[valid_time].astype("int64").to_numpy(dtype=np.float64)
            / 1e9
        )
        for feature, (source_column, lag_hours) in LAG_FEATURE_SPECS.items():
            frame[feature] = interpolate_omni_lag(
                observation_seconds,
                omni.timestamps,
                omni.values[source_column],
                lag_hours,
            )
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
        invalid_rows = int((~finite).sum())
        frame = frame.loc[finite].copy()
        frame["source_row"] = frame.index.to_numpy(dtype=np.int32)

        sampled = False
        if max_rows_per_day > 0 and len(frame) > max_rows_per_day:
            # 每日使用不同但可复现的种子，避免所有日期抽到相同位置模式。
            frame = frame.sample(
                n=max_rows_per_day,
                random_state=sampling_seed + doy,
                replace=False,
            ).sort_index()
            sampled = True
        frame["source_doy"] = np.int16(doy)
        parts.append(frame)
        daily_rows.append(
            {
                "doy": doy,
                "source_file": path.name,
                "original_rows": original_rows,
                "invalid_rows_removed": invalid_rows,
                "rows_after_sampling": len(frame),
                "sampled": sampled,
            }
        )
        if position % 50 == 0 or position == len(files):
            print(f"已读取 {position}/{len(files)} 个年积日文件", flush=True)

    data = pd.concat(parts, ignore_index=True)
    if data.empty:
        raise ValueError("清洗后没有可用于训练的数据")
    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        data[column] = data[column].astype(np.float32)
    data["source_doy"] = data["source_doy"].astype(np.int16)
    data["source_row"] = data["source_row"].astype(np.int32)
    return data, pd.DataFrame(daily_rows)


def allocate_day_counts(day_count: int, ratios: np.ndarray) -> np.ndarray:
    """按比例分配日数，前两组向下取整，剩余日期归入测试集。"""

    counts = np.empty(3, dtype=int)
    counts[0] = int(np.floor(ratios[0] * day_count))
    counts[1] = int(np.floor(ratios[1] * day_count))
    counts[2] = day_count - counts[0] - counts[1]
    if np.any(counts == 0):
        raise ValueError(f"{day_count} 个年积日不足以按当前比例建立三个非空集合")
    return counts


def split_complete_doys(
    data: pd.DataFrame,
    daily: pd.DataFrame,
    ratios: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """随机分配完整年积日，并验证同一天绝不跨集合。"""

    days = np.sort(data["source_doy"].unique())
    shuffled = np.random.default_rng(seed).permutation(days)
    counts = allocate_day_counts(len(days), ratios)
    assignments: list[dict[str, Any]] = []
    start = 0
    for split, count in zip(SPLIT_ORDER, counts):
        for doy in np.sort(shuffled[start : start + count]):
            assignments.append({"doy": int(doy), "split": split})
        start += count
    day_assignments = pd.DataFrame(assignments).sort_values("doy").reset_index(drop=True)
    day_assignments = day_assignments.merge(daily, on="doy", how="left", validate="one_to_one")
    lookup = day_assignments.set_index("doy")["split"]
    row_split = data["source_doy"].map(lookup)
    if row_split.isna().any():
        raise RuntimeError("部分记录没有年积日集合归属")
    if day_assignments.groupby("doy")["split"].nunique().max() != 1:
        raise RuntimeError("至少一个年积日同时属于多个集合")
    split_indices = {
        split: np.flatnonzero(row_split.to_numpy() == split) for split in SPLIT_ORDER
    }
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise RuntimeError("产生了空的数据集合")
    return split_indices, day_assignments


def make_group_kfold_indices(
    data: pd.DataFrame,
    development_indices: np.ndarray,
    n_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[int, int]]:
    """在开发集内部按完整 DOY 建立 GroupKFold，并返回绝对行号。"""

    development_days = data.iloc[development_indices]["source_doy"].to_numpy()
    unique_days = np.unique(development_days)
    if n_splits > len(unique_days):
        raise ValueError(
            f"cv-folds={n_splits} 超过开发集年积日数量 {len(unique_days)}"
        )
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    validation_fold_by_day: dict[int, int] = {}
    placeholder = np.zeros((len(development_indices), 1), dtype=np.int8)
    for fold_number, (train_relative, validation_relative) in enumerate(
        splitter.split(placeholder, groups=development_days), start=1
    ):
        train_indices = development_indices[train_relative]
        validation_indices = development_indices[validation_relative]
        train_days = set(data.iloc[train_indices]["source_doy"].astype(int))
        validation_days = set(data.iloc[validation_indices]["source_doy"].astype(int))
        if train_days & validation_days:
            raise RuntimeError(f"第 {fold_number} 折出现 DOY 泄漏")
        for doy in validation_days:
            if doy in validation_fold_by_day:
                raise RuntimeError(f"DOY {doy:03d} 被分配到多个验证折")
            validation_fold_by_day[doy] = fold_number
        folds.append((train_indices, validation_indices))
    if set(map(int, unique_days)) != set(validation_fold_by_day):
        raise RuntimeError("并非所有开发集 DOY 都恰好作为一次验证折")
    return folds, validation_fold_by_day


def model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """构造固定的、带多重正则化的 XGBoost 参数。"""

    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "max_bin": args.max_bin,
        "tree_method": "hist",
        "device": args.device,
        "random_state": args.model_seed,
        "n_jobs": args.n_jobs,
        "verbosity": 1,
    }


def build_tail_sample_weights(
    y_train: np.ndarray,
    tail_quantile: float,
    tail_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """仅用训练目标的两端分位数构造样本权重，避免数据泄漏。

    residual 小于等于下分位点或大于等于上分位点时使用 ``tail_weight``，
    其余样本权重为 1。验证集和测试集不参与阈值计算，也不进行加权。
    """

    lower_threshold, upper_threshold = np.quantile(
        y_train, [tail_quantile, 1.0 - tail_quantile]
    )
    if not lower_threshold < upper_threshold:
        raise ValueError("训练集 residual 分布不足以建立互不重叠的两端区间")

    lower_mask = y_train <= lower_threshold
    upper_mask = y_train >= upper_threshold
    tail_mask = lower_mask | upper_mask
    weights = np.ones(len(y_train), dtype=np.float32)
    weights[tail_mask] = np.float32(tail_weight)
    summary = {
        "method": "train_target_two_sided_quantile",
        "tail_quantile_each_side": float(tail_quantile),
        "lower_threshold": float(lower_threshold),
        "upper_threshold": float(upper_threshold),
        "center_weight": 1.0,
        "tail_weight": float(tail_weight),
        "training_rows": int(len(y_train)),
        "lower_tail_rows": int(lower_mask.sum()),
        "upper_tail_rows": int(upper_mask.sum()),
        "total_tail_rows": int(tail_mask.sum()),
        "total_tail_fraction": float(tail_mask.mean()),
    }
    return weights, summary


def fit_model(
    parameters: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_sample_weight: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    early_stopping_rounds: int,
) -> xgb.XGBRegressor:
    """使用训练样本权重，并兼容不同版本的验证集早停写法。"""

    if early_stopping_rounds <= 0:
        model = xgb.XGBRegressor(**parameters)
        model.fit(
            x_train,
            y_train,
            sample_weight=train_sample_weight,
            eval_set=[(x_train, y_train), (x_validation, y_validation)],
            verbose=False,
        )
        return model
    try:
        model = xgb.XGBRegressor(
            **parameters, early_stopping_rounds=early_stopping_rounds
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=train_sample_weight,
            eval_set=[(x_train, y_train), (x_validation, y_validation)],
            verbose=False,
        )
    except TypeError:
        model = xgb.XGBRegressor(**parameters)
        model.fit(
            x_train,
            y_train,
            sample_weight=train_sample_weight,
            eval_set=[(x_train, y_train), (x_validation, y_validation)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    return model


def calculate_metrics(
    true: np.ndarray, predicted: np.ndarray, day_count: int
) -> SplitMetrics:
    """计算回归性能、偏差和相关系数。"""

    correlation = (
        float(np.corrcoef(true, predicted)[0, 1])
        if len(true) > 1 and np.std(true) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return SplitMetrics(
        rows=len(true),
        days=day_count,
        rmse=float(np.sqrt(mean_squared_error(true, predicted))),
        mae=float(mean_absolute_error(true, predicted)),
        r2=float(r2_score(true, predicted)),
        bias=float(np.mean(predicted - true)),
        correlation=correlation,
    )


def feature_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    """导出权重和增益两类特征重要性。"""

    booster = model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    weight_raw = booster.get_score(importance_type="weight")
    rows = []
    for index, feature in enumerate(FEATURE_COLUMNS):
        key = f"f{index}"
        rows.append(
            {
                "feature": feature,
                "gain": float(gain_raw.get(key, 0.0)),
                "weight": float(weight_raw.get(key, 0.0)),
                "sklearn_importance": float(model.feature_importances_[index]),
            }
        )
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def select_plot_sample(size: int, maximum: int, seed: int) -> np.ndarray:
    """为绘图选择可复现的随机子样本，不影响训练和评价。"""

    if size <= maximum:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, maximum, replace=False))


def plot_learning_curve(
    histories: list[dict[str, dict[str, list[float]]]],
    best_iterations: list[int],
    output: Path,
    dpi: int,
) -> None:
    """绘制三折训练/验证 RMSE，并标出每一折的最佳轮次。"""

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(histories)))
    for fold_number, (history, best_iteration, color) in enumerate(
        zip(histories, best_iterations, colors), start=1
    ):
        train_rmse = history["validation_0"]["rmse"]
        validation_rmse = history["validation_1"]["rmse"]
        axis.plot(
            train_rmse,
            color=color,
            alpha=0.45,
            linestyle="--",
            label=f"Fold {fold_number} train",
        )
        axis.plot(
            validation_rmse,
            color=color,
            linewidth=1.5,
            label=f"Fold {fold_number} validation",
        )
        axis.axvline(best_iteration, color=color, alpha=0.45, linestyle=":")
    axis.set(
        xlabel="Boosting round",
        ylabel="RMSE (TECU)",
        title="Three-fold GroupKFold learning curves",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_test_scatter(
    true: np.ndarray,
    predicted: np.ndarray,
    output: Path,
    maximum_points: int,
    seed: int,
    dpi: int,
) -> None:
    """绘制测试集密度图，并标注基于完整测试集计算的 RMSE 和 R²。"""

    rmse = float(np.sqrt(mean_squared_error(true, predicted)))
    r2 = float(r2_score(true, predicted))
    chosen = select_plot_sample(len(true), maximum_points, seed)
    x, y = true[chosen], predicted[chosen]
    lower = float(min(np.quantile(x, 0.001), np.quantile(y, 0.001)))
    upper = float(max(np.quantile(x, 0.999), np.quantile(y, 0.999)))
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.hexbin(x, y, gridsize=100, mincnt=1, bins="log", cmap="viridis")
    axis.plot([lower, upper], [lower, upper], "r--", linewidth=1)
    axis.set(
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="True residual (TECU)",
        ylabel="Predicted residual (TECU)",
        title="Test set: true vs predicted residual",
    )
    axis.text(
        0.03,
        0.97,
        f"RMSE = {rmse:.4f} TECU\n$R^2$ = {r2:.4f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    figure.colorbar(image, ax=axis, label="log10(count)")
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_target_distribution(
    targets: dict[str, np.ndarray], output: Path, dpi: int
) -> None:
    """比较开发集和独立测试集目标分布。"""

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    combined = np.concatenate(list(targets.values()))
    lower, upper = np.quantile(combined, [0.001, 0.999])
    bins = np.linspace(lower, upper, 100)
    for split, values in targets.items():
        values = values[(values >= lower) & (values <= upper)]
        axis.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=split)
    axis.set(
        xlabel="Residual (TECU)",
        ylabel="Density",
        title="Target distribution by complete-DOY outer split",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def plot_importance(importance: pd.DataFrame, output: Path, dpi: int) -> None:
    """绘制按增益排序的特征重要性。"""

    ordered = importance.sort_values("gain", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.barh(ordered["feature"], ordered["gain"], color="#2878b5")
    axis.set(xlabel="Gain", title="XGBoost feature importance")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def json_ready(value: Any) -> Any:
    """把 NumPy/Path 类型转换为 JSON 可序列化值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> None:
    """执行数据读取、完整 DOY 划分、训练、评价和结果导出。"""

    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.input_dir, args.year)
    print(f"发现 {len(files)} 个年积日文件", flush=True)
    print(f"目标：{TARGET_COLUMN}", flush=True)
    print(f"特征（不含目标）：{', '.join(FEATURE_COLUMNS)}", flush=True)
    data, daily = load_data(files, args.max_rows_per_day, args.sampling_seed)

    ratios = np.array(
        [args.train_ratio, args.validation_ratio, args.test_ratio], dtype=float
    )
    split_indices, day_assignments = split_complete_doys(
        data, daily, ratios, args.split_seed
    )
    day_assignments.to_csv(
        args.output_dir / "doy_split_assignments.csv", index=False, encoding="utf-8-sig"
    )

    # 明确验证每个 DOY 只出现一次，作为防泄漏的硬约束。
    if day_assignments["doy"].duplicated().any():
        raise RuntimeError("DOY 划分表存在重复日期")
    for split in SPLIT_ORDER:
        indices = split_indices[split]
        split_days = day_assignments.loc[day_assignments["split"].eq(split), "doy"]
        print(
            f"{split}: {len(split_days)} 天，{len(indices):,} 行，"
            f"DOY 范围 {split_days.min():03d}–{split_days.max():03d}",
            flush=True,
        )

    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    target = data[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
    matrices = {
        split: features[indices] for split, indices in split_indices.items()
    }
    targets = {split: target[indices] for split, indices in split_indices.items()}
    train_sample_weight, tail_weighting = build_tail_sample_weights(
        targets["train"], args.tail_quantile, args.tail_weight
    )

    params = model_parameters(args)
    print(
        f"开始训练：最多 {args.n_estimators} 轮，早停 {args.early_stopping_rounds} 轮，"
        f"L1={args.reg_alpha:g}，L2={args.reg_lambda:g}",
        flush=True,
    )
    print(
        "训练集两端加权："
        f"residual <= {tail_weighting['lower_threshold']:.4f} 或 "
        f">= {tail_weighting['upper_threshold']:.4f} 时权重="
        f"{tail_weighting['tail_weight']:g}，共 "
        f"{tail_weighting['total_tail_rows']:,} 行 "
        f"({tail_weighting['total_tail_fraction']:.1%})",
        flush=True,
    )
    model = fit_model(
        params,
        matrices["train"],
        targets["train"],
        train_sample_weight,
        matrices["validation"],
        targets["validation"],
        args.early_stopping_rounds,
    )
    model.save_model(args.output_dir / "xg_cosmic_residual_model.json")

    predictions = {
        split: model.predict(matrices[split]).astype(np.float32)
        for split in SPLIT_ORDER
    }
    metrics: dict[str, SplitMetrics] = {}
    prediction_frames: list[pd.DataFrame] = []
    for split in SPLIT_ORDER:
        indices = split_indices[split]
        day_count = int(day_assignments["split"].eq(split).sum())
        metrics[split] = calculate_metrics(
            targets[split], predictions[split], day_count
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "source_doy": data.iloc[indices]["source_doy"].to_numpy(),
                    "source_row": data.iloc[indices]["source_row"].to_numpy(),
                    "true_residual": targets[split],
                    "predicted_residual": predictions[split],
                    "prediction_error": predictions[split] - targets[split],
                }
            )
        )
        result = metrics[split]
        print(
            f"{split}: RMSE={result.rmse:.4f}, MAE={result.mae:.4f}, "
            f"R2={result.r2:.4f}, bias={result.bias:.4f}",
            flush=True,
        )

    metric_frame = pd.DataFrame(
        [{"split": split, **asdict(metrics[split])} for split in SPLIT_ORDER]
    )
    metric_frame.to_csv(
        args.output_dir / "split_metrics.csv", index=False, encoding="utf-8-sig"
    )
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(
        args.output_dir / "all_split_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )

    daily_test = (
        prediction_frames[2]
        .assign(squared_error=lambda frame: frame["prediction_error"] ** 2)
        .groupby("source_doy", as_index=False)
        .agg(
            rows=("true_residual", "size"),
            true_mean=("true_residual", "mean"),
            predicted_mean=("predicted_residual", "mean"),
            bias=("prediction_error", "mean"),
            mae=("prediction_error", lambda values: np.mean(np.abs(values))),
            mean_squared_error=("squared_error", "mean"),
        )
    )
    daily_test["rmse"] = np.sqrt(daily_test.pop("mean_squared_error"))
    daily_test.to_csv(
        args.output_dir / "test_daily_metrics.csv", index=False, encoding="utf-8-sig"
    )

    importance = feature_importance(model)
    importance.to_csv(
        args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    history = model.evals_result()
    history_frame = pd.DataFrame(
        {
            "round": np.arange(len(history["validation_0"]["rmse"])),
            "train_rmse": history["validation_0"]["rmse"],
            "validation_rmse": history["validation_1"]["rmse"],
        }
    )
    history_frame.to_csv(
        args.output_dir / "learning_curve.csv", index=False, encoding="utf-8-sig"
    )

    plot_learning_curve(model, args.output_dir / "learning_curve.png", args.dpi)
    plot_test_scatter(
        targets["test"],
        predictions["test"],
        args.output_dir / "test_true_vs_predicted.png",
        args.max_plot_points,
        args.model_seed,
        args.dpi,
    )
    plot_target_distribution(
        targets, args.output_dir / "target_distribution_by_split.png", args.dpi
    )
    plot_importance(
        importance, args.output_dir / "feature_importance.png", args.dpi
    )

    run_metadata = {
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "target_leakage_prevention": "residual excluded from features",
        "split_unit": "complete source DOY",
        "split_ratios": dict(zip(SPLIT_ORDER, ratios.tolist())),
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "sampling_seed": args.sampling_seed,
        "model_parameters": params,
        "tail_weighting": tail_weighting,
        "best_iteration_zero_based": getattr(model, "best_iteration", None),
        "best_score": getattr(model, "best_score", None),
        "metrics": {split: asdict(value) for split, value in metrics.items()},
        "arguments": vars(args),
        "library_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, default=json_ready),
        encoding="utf-8",
    )
    print(f"训练完成，结果目录：{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
