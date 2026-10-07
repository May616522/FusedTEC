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
import gc
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
import optuna
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
        "--optuna-trials",
        type=int,
        default=20,
        help="Optuna 参数搜索次数；每次都执行完整 GroupKFold",
    )
    parser.add_argument(
        "--optuna-timeout-minutes",
        type=float,
        default=0.0,
        help="Optuna 最长搜索分钟数；0 表示仅由 trials 控制",
    )
    parser.add_argument("--optuna-seed", type=int, default=42)
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
    if args.optuna_trials < 1:
        raise ValueError("optuna-trials 必须大于 0")
    if args.optuna_timeout_minutes < 0.0:
        raise ValueError("optuna-timeout-minutes 不能为负")
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


def suggest_model_parameters(
    trial: optuna.Trial, base_parameters: dict[str, Any]
) -> dict[str, Any]:
    """在保留固定轮数和设备设置的前提下生成一组 XGBoost 候选参数。"""

    parameters = dict(base_parameters)
    parameters.update(
        {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.08, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 2.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.70, 1.00
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.5, 10.0, log=True
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        }
    )
    return parameters


def optimize_parameters(
    base_parameters: dict[str, Any],
    features: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> optuna.Study:
    """以三折完整-DOY合并 RMSE 为目标执行 Optuna 搜索。"""

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_model_parameters(trial, base_parameters)
        total_squared_error = 0.0
        total_rows = 0
        fold_rmse: list[float] = []
        fold_best_iterations: list[int] = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            folds, start=1
        ):
            weights, _ = build_tail_sample_weights(
                target[train_indices], args.tail_quantile, args.tail_weight
            )
            model = fit_model(
                parameters,
                features[train_indices],
                target[train_indices],
                weights,
                features[validation_indices],
                target[validation_indices],
                args.early_stopping_rounds,
            )
            predicted = model.predict(features[validation_indices]).astype(np.float32)
            errors = predicted - target[validation_indices]
            total_squared_error += float(np.dot(errors, errors))
            total_rows += len(errors)
            current_rmse = float(np.sqrt(np.mean(errors * errors)))
            fold_rmse.append(current_rmse)
            fold_best_iterations.append(int(getattr(model, "best_iteration", 0)))
            combined_rmse = float(np.sqrt(total_squared_error / total_rows))
            trial.report(combined_rmse, step=fold_number)
            del model, predicted, errors, weights
            gc.collect()
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_rmse", fold_rmse)
        trial.set_user_attr("fold_best_iterations_zero_based", fold_best_iterations)
        return float(np.sqrt(total_squared_error / total_rows))

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, args.optuna_trials), n_warmup_steps=1
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="cosmic_residual_complete_doy_groupkfold",
    )

    progress_path = args.output_dir / "optuna_trials.csv"

    def save_progress(current_study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        """每个试验结束后保存进度，避免长作业中断时完全丢失记录。"""

        current_study.trials_dataframe().to_csv(
            progress_path, index=False, encoding="utf-8-sig"
        )

    timeout_seconds = (
        None
        if args.optuna_timeout_minutes <= 0.0
        else args.optuna_timeout_minutes * 60.0
    )
    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=timeout_seconds,
        callbacks=[save_progress],
        gc_after_trial=True,
        show_progress_bar=False,
    )
    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        raise RuntimeError("Optuna 没有完成任何可用试验")
    study.trials_dataframe().to_csv(
        progress_path, index=False, encoding="utf-8-sig"
    )
    return study


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


def plot_optuna_history(study: optuna.Study, output: Path, dpi: int) -> None:
    """绘制已完成 Optuna 试验的交叉验证 RMSE 和历次最优值。"""

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        return
    numbers = np.array([trial.number for trial in completed], dtype=int)
    values = np.array([trial.value for trial in completed], dtype=float)
    running_best = np.minimum.accumulate(values)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.scatter(numbers, values, s=28, alpha=0.75, label="Completed trial")
    axis.plot(numbers, running_best, color="red", linewidth=1.5, label="Best so far")
    axis.set(
        xlabel="Optuna trial",
        ylabel="Three-fold OOF RMSE (TECU)",
        title="Optuna optimization history",
    )
    axis.grid(alpha=0.25)
    axis.legend()
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
    """生成滞后特征，执行 Optuna+GroupKFold，训练最终模型并导出结果。"""

    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(args.input_dir, args.year)
    omni = load_omni_lag_source(args.omni_file, args.year)
    print(f"发现 {len(files)} 个年积日文件", flush=True)
    print(f"OMNI 滞后数据：{args.omni_file}", flush=True)
    print(f"目标：{TARGET_COLUMN}", flush=True)
    print(f"特征（不含目标）：{', '.join(FEATURE_COLUMNS)}", flush=True)
    data, daily = load_data(
        files, omni, args.max_rows_per_day, args.sampling_seed
    )

    ratios = np.array(
        [args.train_ratio, args.validation_ratio, args.test_ratio], dtype=float
    )
    initial_indices, day_assignments = split_complete_doys(
        data, daily, ratios, args.split_seed
    )
    # 原训练集和验证集合并为开发集；原测试 DOY 保持不变并全程隔离。
    development_indices = np.sort(
        np.concatenate([initial_indices["train"], initial_indices["validation"]])
    )
    test_indices = initial_indices["test"]
    folds, validation_fold_by_day = make_group_kfold_indices(
        data, development_indices, args.cv_folds
    )
    day_assignments["outer_split"] = np.where(
        day_assignments["split"].eq("test"), "test", "development"
    )
    day_assignments["cv_validation_fold"] = (
        day_assignments["doy"].map(validation_fold_by_day).astype("Int64")
    )
    day_assignments.to_csv(
        args.output_dir / "doy_split_assignments.csv", index=False, encoding="utf-8-sig"
    )
    if day_assignments["doy"].duplicated().any():
        raise RuntimeError("DOY 划分表存在重复日期")

    development_days = np.unique(data.iloc[development_indices]["source_doy"])
    test_days = np.unique(data.iloc[test_indices]["source_doy"])
    print(
        f"开发集：{len(development_days)} 天，{len(development_indices):,} 行；"
        f"独立测试集：{len(test_days)} 天，{len(test_indices):,} 行",
        flush=True,
    )
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        print(
            f"CV fold {fold_number}: 训练 "
            f"{data.iloc[train_indices]['source_doy'].nunique()} 天/{len(train_indices):,} 行，"
            f"验证 {data.iloc[validation_indices]['source_doy'].nunique()} 天/"
            f"{len(validation_indices):,} 行",
            flush=True,
        )

    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    target = data[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
    base_parameters = model_parameters(args)
    print(
        f"开始 Optuna：{args.optuna_trials} 次试验，每次 {args.cv_folds} 折，"
        f"每折最多 {args.n_estimators} 轮、早停 {args.early_stopping_rounds} 轮",
        flush=True,
    )
    study = optimize_parameters(base_parameters, features, target, folds, args)
    best_parameters = dict(base_parameters)
    best_parameters.update(study.best_trial.params)
    optuna_summary = {
        "best_trial_number": int(study.best_trial.number),
        "best_complete_doy_cv_rmse": float(study.best_value),
        "best_searched_parameters": study.best_trial.params,
        "completed_trials": int(
            sum(
                trial.state == optuna.trial.TrialState.COMPLETE
                for trial in study.trials
            )
        ),
        "pruned_trials": int(
            sum(
                trial.state == optuna.trial.TrialState.PRUNED
                for trial in study.trials
            )
        ),
    }
    (args.output_dir / "optuna_best_params.json").write_text(
        json.dumps(optuna_summary, ensure_ascii=False, indent=2, default=json_ready),
        encoding="utf-8",
    )
    try:
        parameter_importance = optuna.importance.get_param_importances(study)
        pd.DataFrame(
            parameter_importance.items(), columns=["parameter", "importance"]
        ).to_csv(
            args.output_dir / "optuna_parameter_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
    except (RuntimeError, ValueError):
        parameter_importance = {}
    plot_optuna_history(study, args.output_dir / "optuna_history.png", args.dpi)
    print(
        f"Optuna 最优试验 {study.best_trial.number}："
        f"三折合并 RMSE={study.best_value:.4f}",
        flush=True,
    )
    print(
        "最优搜索参数："
        + ", ".join(f"{key}={value}" for key, value in study.best_trial.params.items()),
        flush=True,
    )

    # 用最优参数重跑三折，得到完整 OOF 预测、学习曲线和稳健的轮次估计。
    oof_predictions = np.full(len(data), np.nan, dtype=np.float32)
    fold_histories: list[dict[str, dict[str, list[float]]]] = []
    fold_best_iterations: list[int] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fold_tail_weighting: list[dict[str, Any]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        weights, weighting_summary = build_tail_sample_weights(
            target[train_indices], args.tail_quantile, args.tail_weight
        )
        model = fit_model(
            best_parameters,
            features[train_indices],
            target[train_indices],
            weights,
            features[validation_indices],
            target[validation_indices],
            args.early_stopping_rounds,
        )
        predicted = model.predict(features[validation_indices]).astype(np.float32)
        oof_predictions[validation_indices] = predicted
        validation_days = int(data.iloc[validation_indices]["source_doy"].nunique())
        fold_metrics = calculate_metrics(
            target[validation_indices], predicted, validation_days
        )
        best_iteration = int(getattr(model, "best_iteration", 0))
        fold_best_iterations.append(best_iteration)
        fold_histories.append(model.evals_result())
        weighting_summary = {"fold": fold_number, **weighting_summary}
        fold_tail_weighting.append(weighting_summary)
        fold_metric_rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train_indices),
                "train_days": int(data.iloc[train_indices]["source_doy"].nunique()),
                "validation_rows": len(validation_indices),
                "validation_days": validation_days,
                "best_iteration_zero_based": best_iteration,
                "best_score": float(getattr(model, "best_score", np.nan)),
                "rmse": fold_metrics.rmse,
                "mae": fold_metrics.mae,
                "r2": fold_metrics.r2,
                "bias": fold_metrics.bias,
                "correlation": fold_metrics.correlation,
            }
        )
        print(
            f"最优参数 fold {fold_number}: RMSE={fold_metrics.rmse:.4f}, "
            f"R2={fold_metrics.r2:.4f}, best_round={best_iteration + 1}",
            flush=True,
        )
        del model, predicted, weights
        gc.collect()

    if not np.isfinite(oof_predictions[development_indices]).all():
        raise RuntimeError("开发集 OOF 预测不完整")
    selected_n_estimators = int(
        np.clip(
            np.rint(np.median(np.asarray(fold_best_iterations) + 1)),
            1,
            args.n_estimators,
        )
    )
    final_weights, final_tail_weighting = build_tail_sample_weights(
        target[development_indices], args.tail_quantile, args.tail_weight
    )
    final_parameters = dict(best_parameters)
    final_parameters["n_estimators"] = selected_n_estimators
    print(
        f"三折最佳轮数：{[value + 1 for value in fold_best_iterations]}；"
        f"最终模型使用中位数 {selected_n_estimators} 轮",
        flush=True,
    )
    final_model = xgb.XGBRegressor(**final_parameters)
    final_model.fit(
        features[development_indices],
        target[development_indices],
        sample_weight=final_weights,
        verbose=False,
    )
    final_model.save_model(args.output_dir / "xg_cosmic_residual_model.json")
    test_predictions = final_model.predict(features[test_indices]).astype(np.float32)

    metrics = {
        "development_oof": calculate_metrics(
            target[development_indices],
            oof_predictions[development_indices],
            len(development_days),
        ),
        "test": calculate_metrics(
            target[test_indices], test_predictions, len(test_days)
        ),
    }
    for split_name, result in metrics.items():
        print(
            f"{split_name}: RMSE={result.rmse:.4f}, MAE={result.mae:.4f}, "
            f"R2={result.r2:.4f}, bias={result.bias:.4f}",
            flush=True,
        )

    development_frame = pd.DataFrame(
        {
            "split": "development_oof",
            "source_doy": data.iloc[development_indices]["source_doy"].to_numpy(),
            "source_row": data.iloc[development_indices]["source_row"].to_numpy(),
            "true_residual": target[development_indices],
            "predicted_residual": oof_predictions[development_indices],
            "prediction_error": (
                oof_predictions[development_indices] - target[development_indices]
            ),
        }
    )
    test_frame = pd.DataFrame(
        {
            "split": "test",
            "source_doy": data.iloc[test_indices]["source_doy"].to_numpy(),
            "source_row": data.iloc[test_indices]["source_row"].to_numpy(),
            "true_residual": target[test_indices],
            "predicted_residual": test_predictions,
            "prediction_error": test_predictions - target[test_indices],
        }
    )
    pd.DataFrame(
        [{"split": split_name, **asdict(value)} for split_name, value in metrics.items()]
    ).to_csv(
        args.output_dir / "split_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([development_frame, test_frame], ignore_index=True).to_csv(
        args.output_dir / "all_split_predictions.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        args.output_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    daily_test = (
        test_frame.assign(squared_error=lambda frame: frame["prediction_error"] ** 2)
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

    importance = feature_importance(final_model)
    importance.to_csv(
        args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )
    learning_frames = []
    for fold_number, (history, best_iteration) in enumerate(
        zip(fold_histories, fold_best_iterations), start=1
    ):
        rounds = len(history["validation_0"]["rmse"])
        learning_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "round": np.arange(rounds),
                    "train_rmse": history["validation_0"]["rmse"],
                    "validation_rmse": history["validation_1"]["rmse"],
                    "best_iteration_zero_based": best_iteration,
                }
            )
        )
    pd.concat(learning_frames, ignore_index=True).to_csv(
        args.output_dir / "learning_curve.csv", index=False, encoding="utf-8-sig"
    )

    plot_learning_curve(
        fold_histories,
        fold_best_iterations,
        args.output_dir / "learning_curve.png",
        args.dpi,
    )
    plot_test_scatter(
        target[test_indices],
        test_predictions,
        args.output_dir / "test_true_vs_predicted.png",
        args.max_plot_points,
        args.model_seed,
        args.dpi,
    )
    plot_target_distribution(
        {
            "development": target[development_indices],
            "test": target[test_indices],
        },
        args.output_dir / "target_distribution_by_split.png",
        args.dpi,
    )
    plot_importance(
        importance, args.output_dir / "feature_importance.png", args.dpi
    )

    run_metadata = {
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "lag_features": LAG_FEATURE_SPECS,
        "omni_file": args.omni_file,
        "target_leakage_prevention": "residual excluded from features",
        "outer_test_policy": "original complete-DOY test split kept fully isolated",
        "cross_validation": {
            "method": "GroupKFold",
            "group": "source_doy",
            "n_splits": args.cv_folds,
            "fold_metrics": fold_metric_rows,
            "selected_n_estimators_from_median_best_round": selected_n_estimators,
        },
        "initial_split_ratios": dict(zip(SPLIT_ORDER, ratios.tolist())),
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "sampling_seed": args.sampling_seed,
        "optuna": {**optuna_summary, "parameter_importance": parameter_importance},
        "base_model_parameters": base_parameters,
        "final_model_parameters": final_parameters,
        "fold_tail_weighting": fold_tail_weighting,
        "final_tail_weighting": final_tail_weighting,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "arguments": vars(args),
        "library_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "optuna": optuna.__version__,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, default=json_ready),
        encoding="utf-8",
    )
    print(f"训练完成，结果目录：{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
