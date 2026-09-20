#!/usr/bin/env python3
"""Tune, train and audit a direct GIM VTEC XGBoost model.

The default workflow keeps statistically rare targets and randomly assigns
complete calendar days to train/validation/test. A calendar day is an
indivisible group: all rows from that day must stay in exactly one split.
The target is ``gim_vtec``. ``TEC_smooth`` is an input feature together with
the existing spatial, temporal and space-weather predictors. Both ``gim_vtec``
and the algebraically related ``residual`` are forbidden as model features.
Only rows satisfying the documented hard-QC rule ``residual > 0`` are retained
before sampling and splitting.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COLUMN = "gim_vtec"
FILTER_COLUMN = "residual"
SOURCE_FILE_COLUMN = "__source_file_id"
SOURCE_ROW_COLUMN = "__source_row_number"
ORIGINAL_ROW_COLUMN = "__cleaned_row_index"
LOCAL_TIME_SIN_COLUMN = "local_time_s"
LOCAL_TIME_COS_COLUMN = "local_time_c"
EXCLUDED_COLUMNS = {
    "datetime", TARGET_COLUMN, FILTER_COLUMN, "TEC_raw", SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN, ORIGINAL_ROW_COLUMN, "HOD_s", "HOD_c",
}
# Compatibility constants for the retained diagnostic helper functions. The
# simplified main workflow below does not invoke sigma/tail segmentation or
# sample weighting.
SIGMA_SEGMENTS = ("within_1sigma", "sigma_1_to_2", "sigma_2_to_3", "beyond_3sigma")
TAIL_SEGMENTS = ("extreme_low", "low", "central", "high", "extreme_high")
WEIGHT_LEVELS = {
    "none": (1.0, 1.0, 1.0),
    "1-2-4": (1.0, 2.0, 4.0),
    "1-2-6": (1.0, 2.0, 6.0),
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括输入/输出目录、Optuna试验数量、XGBoost参数范围等。"""
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-tuned XGBoost regression for direct GIM VTEC prediction with a "
            "leakage-safe, calendar-day-grouped train/validation/test split."
        )
    )
    parser.add_argument(
        "--input-dir", type=Path,
        help="Input CSV directory; not required with --evaluation-only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-only", action="store_true",
        help=(
            "Do not train. Recalculate direct GIM prediction metrics from an "
            "existing output directory produced by this workflow."
        ),
    )
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--csv-engine", choices=("c", "python", "pyarrow"), default="c")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--optuna-timeout", type=int, default=3600,
        help="Optuna time limit in seconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300,
        help="Maximum boosting rounds; default: 300.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument(
        "--reg-alpha", type=float, default=0.1,
        help="Positive L1 regularization coefficient; default: 0.1.",
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0,
        help="Positive L2 regularization coefficient; default: 1.0.",
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--sample-fraction", type=float, default=1.0,
        help=(
            "Randomly retain this fraction within every valid calendar day. "
            "The default 1.0 retains every valid row; smaller values keep at "
            "least one row from every day."
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed used only for --sample-fraction; default: 42.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed used only by --split-strategy random-day.",
    )
    parser.add_argument(
        "--split-strategy", choices=("random-day", "time"), default="random-day",
        help=(
            "Randomly assign indivisible calendar days (default), or use "
            "chronological whole-day blocks for an optional extrapolation test."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--fixed-altitude", type=float, default=None,
        help="Replace alt by this constant; default is the cleaned-data median.",
    )
    parser.add_argument(
        "--n-jobs", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
    )
    parser.add_argument("--read-batch-size", type=int, default=250)
    parser.add_argument("--max-scatter-points", type=int, default=200_000)
    parser.add_argument("--save-eda", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    """打印带时间戳的日志信息。"""
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def validate_args(args: argparse.Namespace) -> None:
    """验证命令行参数的有效性，确保数值范围合理。"""
    if not args.evaluation_only and args.input_dir is None:
        raise ValueError("--input-dir is required unless --evaluation-only is used")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if args.optuna_timeout < 0:
        raise ValueError("--optuna-timeout cannot be negative")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be positive")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds cannot be negative")
    if not np.isfinite(args.sample_fraction) or not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in the interval (0, 1]")
    if not np.isfinite(args.reg_alpha) or args.reg_alpha <= 0:
        raise ValueError("--reg-alpha must be a positive finite number")
    if not np.isfinite(args.reg_lambda) or args.reg_lambda <= 0:
        raise ValueError("--reg-lambda must be a positive finite number")
    split_ratios = np.array(
        [args.train_ratio, args.validation_ratio, args.test_ratio], dtype=float
    )
    if not np.all(np.isfinite(split_ratios)) or np.any(split_ratios <= 0):
        raise ValueError("All split ratios must be positive finite numbers")
    if not np.isclose(split_ratios.sum(), 1.0):
        raise ValueError("Train/validation/test ratios must sum to 1.0")
    if args.n_jobs < 1 or args.read_batch_size < 1:
        raise ValueError("--n-jobs and --read-batch-size must be positive")


def find_csv_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    """在指定目录中查找匹配模式的CSV文件。"""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {input_dir}")
    return files


def load_csv_files(files: list[Path], csv_engine: str, batch_size: int) -> pd.DataFrame:
    """批量读取多个CSV文件并整合为单个DataFrame，保留数据源信息。"""
    batches: list[pd.DataFrame] = []
    current: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for number, path in enumerate(files, start=1):
        frame = pd.read_csv(path, engine=csv_engine)
        columns = frame.columns.tolist()
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                "CSV columns/order differ: "
                f"first file={expected_columns}; {path}={columns}"
            )
        frame[SOURCE_FILE_COLUMN] = number - 1
        frame[SOURCE_ROW_COLUMN] = np.arange(len(frame), dtype=np.int64)
        current.append(frame)
        if len(current) >= batch_size:
            batches.append(pd.concat(current, ignore_index=True))
            current.clear()
        if number % 500 == 0 or number == len(files):
            log(f"Read {number}/{len(files)} CSV files")
    if current:
        batches.append(pd.concat(current, ignore_index=True))
    data = pd.concat(batches, ignore_index=True)
    del batches
    gc.collect()
    return data


def sample_rows_within_days(
    data: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """在每个自然日内部独立随机抽样，确保所有日期均有记录。"""
    rows_before = len(data)
    calendar_dates = data["datetime"].dt.normalize()
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    daily_rows: list[dict[str, object]] = []
    for calendar_date, positions in calendar_dates.groupby(calendar_dates).groups.items():
        positions_array = np.asarray(positions, dtype=np.int64)
        sample_size = len(positions_array)
        if fraction < 1.0:
            sample_size = max(1, int(round(len(positions_array) * fraction)))
            sample_size = min(sample_size, len(positions_array))
            selected = rng.choice(positions_array, size=sample_size, replace=False)
        else:
            selected = positions_array
        selected_parts.append(selected)
        daily_rows.append({
            "calendar_date": calendar_date,
            "year": int(calendar_date.year),
            "day_of_year": int(calendar_date.dayofyear),
            "month": int(calendar_date.month),
            "rows_before_sampling": int(len(positions_array)),
            "rows_after_sampling": int(sample_size),
            "realized_fraction": float(sample_size / len(positions_array)),
        })
    selected_index = np.sort(np.concatenate(selected_parts))
    sampled = data.iloc[selected_index].reset_index(drop=True)
    daily_sampling = pd.DataFrame(daily_rows).sort_values("calendar_date").reset_index(drop=True)
    sampled = sampled.reset_index(drop=True)
    summary = {
        "requested_fraction": float(fraction),
        "seed": int(seed),
        "rows_before_sampling": int(rows_before),
        "rows_after_sampling": int(len(sampled)),
        "realized_fraction": float(len(sampled) / rows_before),
        "calendar_days_before_sampling": int(calendar_dates.nunique()),
        "calendar_days_after_sampling": int(sampled["datetime"].dt.normalize().nunique()),
        "minimum_rows_retained_per_day": int(daily_sampling["rows_after_sampling"].min()),
        "stage": "within each valid calendar day before whole-day random split",
    }
    log(
        f"Within-day random sampling: retained {len(sampled):,}/{rows_before:,} valid rows "
        f"({len(sampled) / rows_before:.2%}) across all "
        f"{summary['calendar_days_after_sampling']:,} days, seed={seed}"
    )
    return sampled, summary, daily_sampling


def clean_and_validate(
    data: pd.DataFrame, requested_altitude: float | None
) -> tuple[
    pd.DataFrame, list[str], float, dict[str, float], dict[str, object], pd.DataFrame
]:
    """清理缺失/无穷值，仅保留 residual > 0，并固定高度值。"""
    required = {
        TARGET_COLUMN, FILTER_COLUMN, "TEC_smooth", "datetime", "lat", "lon", "alt"
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    # UTC-hour sine/cosine are replaced below by longitude-adjusted local time.
    data = data.drop(columns=["HOD_s", "HOD_c"], errors="ignore")
    parsed_datetime = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
    invalid_datetime_count = int(parsed_datetime.isna().sum())
    if invalid_datetime_count:
        log(f"Rows with invalid datetime values: {invalid_datetime_count:,}")
    data["datetime"] = parsed_datetime

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    rows_before = len(data)
    missing_or_nonfinite = data.isna().any(axis=1)
    nonpositive_residual = data[FILTER_COLUMN] <= 0
    duplicate_columns = [
        column for column in data.columns
        if column not in {SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN}
    ]
    exact_duplicate = data.duplicated(subset=duplicate_columns, keep="first")
    removed_mask = missing_or_nonfinite | exact_duplicate | nonpositive_residual
    removed_records = data.loc[
        removed_mask,
        [SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN, "datetime"],
    ].copy()
    removed_records["qc_reason"] = np.select(
        [nonpositive_residual[removed_mask], missing_or_nonfinite[removed_mask]],
        ["residual_not_positive", "missing_or_nonfinite"],
        default="exact_duplicate",
    )
    data = data.loc[~removed_mask].reset_index(drop=True)
    qc_log: dict[str, object] = {
        "rows_before": int(rows_before),
        "invalid_datetime": invalid_datetime_count,
        "missing_or_nonfinite": int(missing_or_nonfinite.sum()),
        "exact_duplicate": int(exact_duplicate.sum()),
        "residual_not_positive": int(nonpositive_residual.sum()),
        "rows_removed": int(removed_mask.sum()),
        "rows_retained": int(len(data)),
        "rules": [
            "all model fields finite/non-missing",
            "remove exact duplicate records",
            "retain only rows where residual > 0 before sampling and splitting",
        ],
    }
    log(f"Rows before cleaning: {rows_before:,}")
    log(f"Rows removed: {rows_before - len(data):,}; retained: {len(data):,}")
    if len(data) < 10:
        raise ValueError(f"Too few valid rows for modeling: {len(data)}")
    data[ORIGINAL_ROW_COLUMN] = np.arange(len(data), dtype=np.int64)

    original_altitude = {
        "minimum": float(data["alt"].min()),
        "median": float(data["alt"].median()),
        "maximum": float(data["alt"].max()),
    }
    fixed_altitude = (
        original_altitude["median"] if requested_altitude is None else float(requested_altitude)
    )
    if not np.isfinite(fixed_altitude):
        raise ValueError("--fixed-altitude must be finite")
    data["alt"] = np.full(len(data), fixed_altitude, dtype=np.float32)
    utc_hour = (
        data["datetime"].dt.hour.to_numpy(dtype=float)
        + data["datetime"].dt.minute.to_numpy(dtype=float) / 60.0
        + data["datetime"].dt.second.to_numpy(dtype=float) / 3600.0
        + data["datetime"].dt.microsecond.to_numpy(dtype=float) / 3_600_000_000.0
    )
    local_solar_hour = np.mod(
        utc_hour + data["lon"].to_numpy(dtype=float) / 15.0, 24.0
    )
    phase = 2.0 * np.pi * local_solar_hour / 24.0
    data[LOCAL_TIME_SIN_COLUMN] = np.sin(phase)
    data[LOCAL_TIME_COS_COLUMN] = np.cos(phase)
    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions")
    if "TEC_smooth" not in feature_columns:
        raise RuntimeError("TEC_smooth must be present as a model feature")
    leaked_columns = sorted({TARGET_COLUMN, FILTER_COLUMN}.intersection(feature_columns))
    if leaked_columns:
        raise RuntimeError(f"Target-derived columns leaked into features: {leaked_columns}")
    non_numeric = [
        column for column in [*feature_columns, TARGET_COLUMN]
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise TypeError(f"Model columns must be numeric: {non_numeric}")
    log(
        "Original altitude range: "
        f"{original_altitude['minimum']:.6f} to {original_altitude['maximum']:.6f}; "
        f"fixed at {fixed_altitude:.6f}"
    )
    log(
        "Replaced UTC HOD_s/HOD_c with local solar-time features "
        f"{LOCAL_TIME_SIN_COLUMN}/{LOCAL_TIME_COS_COLUMN} computed from UTC datetime and longitude"
    )
    return (
        data, feature_columns, fixed_altitude, original_altitude,
        qc_log, removed_records,
    )


def build_sigma_filtered_splits(
    data: pd.DataFrame, splits: dict[str, np.ndarray], outlier_sigma: float,
    apply_filter: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """仅用训练集计算目标 sigma 阈值；可选过滤仅作用于训练集。"""
    train_target = data.iloc[splits["train"]][TARGET_COLUMN].to_numpy()
    target_mean = float(np.mean(train_target))
    target_std = float(np.std(train_target, ddof=0))
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Training target mean/std is not finite or std is zero")

    lower_bound = target_mean - outlier_sigma * target_std
    upper_bound = target_mean + outlier_sigma * target_std
    target = data[TARGET_COLUMN].to_numpy()
    clean_mask = (target >= lower_bound) & (target <= upper_bound)

    filtered_splits = {
        "train": (
            splits["train"][clean_mask[splits["train"]]]
            if apply_filter else splits["train"]
        ),
        # 验证集和测试集始终保持自然目标分布。
        "validation": splits["validation"],
        "test": splits["test"],
    }
    test_clean_index = splits["test"][clean_mask[splits["test"]]]
    if len(filtered_splits["train"]) == 0:
        raise ValueError("Sigma filtering produced an empty training split")
    if len(test_clean_index) == 0:
        raise ValueError("Training-derived sigma bounds produced an empty clean test subset")

    split_summary: dict[str, dict[str, float | int]] = {}
    for name, split_index in splits.items():
        retained = int(clean_mask[split_index].sum())
        total = len(split_index)
        split_summary[name] = {
            "rows_before": total,
            "rows_within_bounds": retained,
            "rows_outside_bounds": total - retained,
            "within_bounds_ratio": retained / total,
        }
        action = (
            "retained without filtering"
            if name != "train" or not apply_filter
            else "within bounds and retained for modeling"
        )
        log(
            f"{name} sigma summary: {retained:,}/{total:,} within bounds; "
            f"{action}"
        )

    sigma_filter: dict[str, object] = {
        "fit_split": "train",
        "sigma_multiplier": float(outlier_sigma),
        "mean": target_mean,
        "population_std": target_std,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "diagnostic_bounds": {
            f"{level}_sigma": {
                "lower": float(target_mean - level * target_std),
                "upper": float(target_mean + level * target_std),
            }
            for level in (1, 2, 3)
        },
        "filter_enabled": bool(apply_filter),
        "application": {
            "train": "filtered" if apply_filter else "not filtered",
            "validation": "never filtered; natural distribution",
            "test": "not filtered; clean subset only marked for additional evaluation",
        },
        "splits": split_summary,
    }
    log(
        f"Training-only target {outlier_sigma:g}-sigma bounds: "
        f"mean={target_mean:.6f}, std={target_std:.6f}, "
        f"bounds=[{lower_bound:.6f}, {upper_bound:.6f}]"
    )
    return filtered_splits, test_clean_index, sigma_filter


def split_by_calendar_day(
    data: pd.DataFrame, daily_sampling: pd.DataFrame, strategy: str, split_seed: int,
    train_ratio: float, validation_ratio: float, test_ratio: float,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """按完整自然日划分；默认随机分配日期，同一自然日绝不跨集合。"""
    date_frame = daily_sampling[[
        "calendar_date", "year", "day_of_year", "month",
        "rows_before_sampling", "rows_after_sampling", "realized_fraction",
    ]].copy()
    if len(date_frame) < 7:
        raise ValueError("Too few calendar days for a grouped train/validation/test split")
    date_frame = date_frame.sort_values("calendar_date").reset_index(drop=True)
    day_count = len(date_frame)
    train_count = int(np.floor(day_count * train_ratio))
    validation_count = int(np.floor(day_count * validation_ratio))
    train_count = min(max(train_count, 1), day_count - 2)
    validation_count = min(max(validation_count, 1), day_count - train_count - 1)
    if strategy == "random-day":
        order = np.random.default_rng(split_seed).permutation(day_count)
        ordered = date_frame.iloc[order].reset_index(drop=True)
    else:
        ordered = date_frame
    train_days = ordered.iloc[:train_count]
    validation_days = ordered.iloc[train_count:train_count + validation_count]
    test_days = ordered.iloc[train_count + validation_count:]

    assignments = []
    for split_name, split_days in (
        ("train", train_days), ("validation", validation_days), ("test", test_days)
    ):
        part = split_days.copy()
        part["split"] = split_name
        assignments.append(part)
    day_assignments = (
        pd.concat(assignments, ignore_index=True)
        .sort_values("calendar_date")
        .reset_index(drop=True)
    )
    split_lookup = day_assignments.set_index("calendar_date")["split"]
    row_split = data["datetime"].dt.normalize().map(split_lookup)
    if row_split.isna().any():
        raise RuntimeError(f"{int(row_split.isna().sum()):,} rows lack a day assignment")
    splits = {
        name: np.flatnonzero(row_split.to_numpy() == name)
        for name in ("train", "validation", "test")
    }
    empty_splits = [name for name, index in splits.items() if len(index) == 0]
    if empty_splits:
        raise ValueError(
            "Calendar-day split produced empty datasets: " + ", ".join(empty_splits)
        )

    total = len(data)
    assigned_rows = sum(len(index) for index in splits.values())
    if assigned_rows != total:
        raise RuntimeError(f"DOY split assigned {assigned_rows:,}/{total:,} rows")

    row_assignments = pd.DataFrame({
        "original_row_index": data[ORIGINAL_ROW_COLUMN].to_numpy(),
        "source_file_id": data[SOURCE_FILE_COLUMN].to_numpy(),
        "source_row_number": data[SOURCE_ROW_COLUMN].to_numpy(),
        "calendar_date": data["datetime"].dt.normalize(),
        "split": row_split,
    })
    if row_assignments.groupby("calendar_date")["split"].nunique().max() != 1:
        raise RuntimeError("At least one calendar date was assigned to multiple splits")
    row_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    day_assignments["calendar_date"] = day_assignments["calendar_date"].dt.strftime("%Y-%m-%d")
    day_assignments.to_csv(output_dir / "day_assignments.csv", index=False)
    log(
        f"Dataset split: {strategy} complete-calendar-day assignment with target ratios "
        f"{train_ratio:g}/{validation_ratio:g}/{test_ratio:g}; seed={split_seed}"
    )
    for name, index in splits.items():
        split_dates = data.iloc[index]["datetime"].dt.normalize()
        log(
            f"{name}: rows={len(index):,} ({len(index) / total:.2%}), "
            f"calendar days={split_dates.nunique():,}, "
            f"date coverage={split_dates.min().date()} to {split_dates.max().date()}"
        )
    return splits, day_assignments


def fit_target_diagnostics(y_train_raw: np.ndarray) -> dict[str, object]:
    """只用未筛选的训练目标拟合分位数、sigma 和 IQR 诊断边界。"""
    quantile_values = np.quantile(y_train_raw, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q01, q05, q25, q50, q75, q95, q99 = map(float, quantile_values)
    mean = float(np.mean(y_train_raw))
    std = float(np.std(y_train_raw, ddof=0))
    iqr = q75 - q25
    return {
        "fit_split": "raw train only",
        "n": int(len(y_train_raw)),
        "mean": mean,
        "population_std": std,
        "quantiles": {
            "q01": q01, "q05": q05, "q25": q25, "q50": q50,
            "q75": q75, "q95": q95, "q99": q99,
        },
        "iqr": iqr,
        "iqr_bounds": {"lower": q25 - 1.5 * iqr, "upper": q75 + 1.5 * iqr},
        "sigma_3_bounds": {"lower": mean - 3.0 * std, "upper": mean + 3.0 * std},
    }


def tail_segment_labels(y: np.ndarray, quantiles: dict[str, float]) -> np.ndarray:
    """用训练集固定的 q01/q05/q95/q99 生成五个互斥尾部分段。"""
    return np.select(
        [
            y < quantiles["q01"],
            y < quantiles["q05"],
            y <= quantiles["q95"],
            y <= quantiles["q99"],
        ],
        TAIL_SEGMENTS[:4],
        default=TAIL_SEGMENTS[4],
    )


def make_sample_weights(
    y_train: np.ndarray, quantiles: dict[str, float], scheme: str, cap: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """建立温和训练权重并归一化为均值 1；验证和测试不加权。"""
    central, moderate, extreme = WEIGHT_LEVELS[scheme]
    labels = tail_segment_labels(y_train, quantiles)
    weights = np.full(len(y_train), central, dtype=np.float32)
    weights[np.isin(labels, ("low", "high"))] = moderate
    weights[np.isin(labels, ("extreme_low", "extreme_high"))] = extreme
    np.minimum(weights, cap, out=weights)
    weights /= np.mean(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("Training sample weights must be finite and strictly positive")
    if not np.isclose(float(np.mean(weights)), 1.0, atol=1e-6):
        raise RuntimeError("Normalized training sample weights do not have mean 1")
    counts = {segment: int(np.sum(labels == segment)) for segment in TAIL_SEGMENTS}
    summary = {
        "scheme": scheme,
        "raw_levels": {"central": central, "moderate_tail": moderate, "extreme_tail": extreme},
        "normalization": "divide by training-weight mean",
        "raw_weight_cap": float(cap),
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "mean": float(weights.mean()),
        "counts_by_tail_segment": counts,
    }
    return weights, summary


def base_parameters(args: argparse.Namespace) -> dict[str, object]:
    """生成XGBoost模型的基础参数配置。"""
    return {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "booster": "gbtree", "tree_method": "hist",
        "n_estimators": args.n_estimators,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.model_seed, "n_jobs": args.n_jobs, "verbosity": 0,
    }


def fit_with_early_stopping(
    params: dict[str, object], x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray, rounds: int,
    sample_weight: np.ndarray | None = None,
) -> xgb.XGBRegressor:
    """使用早停策略训练XGBoost回归模型，兼容不同版本的XGBoost API。"""
    eval_set = [(x_train, y_train), (x_validation, y_validation)]
    if rounds == 0:
        model = xgb.XGBRegressor(**params)
        model.fit(
            x_train, y_train, sample_weight=sample_weight,
            eval_set=eval_set, verbose=False,
        )
        return model
    model = xgb.XGBRegressor(**params)
    try:
        model.fit(
            x_train, y_train, sample_weight=sample_weight, eval_set=eval_set,
            early_stopping_rounds=rounds, verbose=False,
        )
    except TypeError:
        # XGBoost >= 2.1 moved early_stopping_rounds into the constructor.
        model = xgb.XGBRegressor(**params, early_stopping_rounds=rounds)
        model.fit(
            x_train, y_train, sample_weight=sample_weight,
            eval_set=eval_set, verbose=False,
        )
    return model


def optimize_parameters(
    args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray,
    sample_weight: np.ndarray | None,
) -> optuna.Study:
    """使用Optuna框架进行XGBoost超参数调优，通过验证集RMSE最小化。"""
    fixed = base_parameters(args)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        }
        model = fit_with_early_stopping(
            params, x_train, y_train, x_validation, y_validation,
            args.early_stopping_rounds, sample_weight,
        )
        prediction = model.predict(x_validation)
        score = float(np.sqrt(mean_squared_error(y_validation, prediction)))
        best_iteration = getattr(model, "best_iteration", params["n_estimators"] - 1)
        trial.set_user_attr("best_iteration", int(best_iteration))
        del model, prediction
        gc.collect()
        return score

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=args.model_seed),
        study_name="xg_jason_gim_vtec",
    )

    def progress(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        if trial.number == 0 or (trial.number + 1) % 10 == 0:
            log(
                f"Optuna trial {trial.number + 1}/{args.n_trials}: "
                f"validation RMSE={trial.value:.6f}; best={study.best_value:.6f}"
            )

    log(
        f"Starting Optuna: n_trials={args.n_trials}, "
        f"timeout={args.optuna_timeout or 'disabled'} seconds"
    )
    study.optimize(
        objective, n_trials=args.n_trials, timeout=args.optuna_timeout or None,
        callbacks=[progress], gc_after_trial=True, show_progress_bar=False,
    )
    log(f"Optuna best validation RMSE: {study.best_value:.6f}")
    return study


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算自然频率下的整体或分段回归指标。"""
    r2 = float("nan")
    pearson = float("nan")
    calibration_slope = float("nan")
    calibration_intercept = float("nan")
    observed_range = float(np.ptp(y_true))
    if len(y_true) >= 2 and not np.isclose(np.var(y_true), 0.0):
        r2 = float(r2_score(y_true, y_pred))
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        calibration_slope, calibration_intercept = map(
            float, np.polyfit(y_true, y_pred, 1)
        )
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "pearson": pearson,
        "bias": float(np.mean(errors)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "absolute_error_p90": float(np.quantile(absolute_errors, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute_errors, 0.95)),
        "error_std": float(np.std(errors)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "mean_difference": float(np.mean(y_pred) - np.mean(y_true)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "range_ratio": float(np.ptp(y_pred) / observed_range) if observed_range else float("nan"),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def evaluate_gim_prediction(
    data: pd.DataFrame,
    split_indices: dict[str, np.ndarray],
    target_values: np.ndarray,
    target_predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """比较 TEC_smooth 直接基线和模型对 GIM VTEC 的预测表现。"""
    expected_target = data[TARGET_COLUMN].to_numpy(dtype=np.float64)
    target_values64 = target_values.astype(np.float64, copy=False)
    maximum_definition_error = float(np.max(np.abs(expected_target - target_values64)))
    if not np.allclose(
        expected_target, target_values64, rtol=1e-5, atol=1e-4
    ):
        raise RuntimeError(
            "Target definition mismatch: expected target = gim_vtec; "
            f"maximum absolute mismatch={maximum_definition_error:.6g}"
        )

    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}
    for split_name, split_index in split_indices.items():
        true_gim = target_values64[split_index]
        tec_smooth_baseline = data.iloc[split_index]["TEC_smooth"].to_numpy(
            dtype=np.float64
        )
        predicted_gim = target_predictions[split_name].astype(
            np.float64, copy=False
        )

        baseline = regression_metrics(true_gim, tec_smooth_baseline)
        model_metrics = regression_metrics(true_gim, predicted_gim)
        improvements = {
            "rmse_absolute_improvement": baseline["rmse"] - model_metrics["rmse"],
            "rmse_improvement_percent": (
                100.0 * (baseline["rmse"] - model_metrics["rmse"]) / baseline["rmse"]
                if baseline["rmse"] else float("nan")
            ),
            "mae_absolute_improvement": baseline["mae"] - model_metrics["mae"],
            "mae_improvement_percent": (
                100.0 * (baseline["mae"] - model_metrics["mae"]) / baseline["mae"]
                if baseline["mae"] else float("nan")
            ),
            "pearson_delta": model_metrics["pearson"] - baseline["pearson"],
            "pearson_improvement_percent": (
                100.0
                * (model_metrics["pearson"] - baseline["pearson"])
                / abs(baseline["pearson"])
                if baseline["pearson"] else float("nan")
            ),
            "r2_delta": model_metrics["r2"] - baseline["r2"],
            "r2_improvement_percent": (
                100.0
                * (model_metrics["r2"] - baseline["r2"])
                / abs(baseline["r2"])
                if baseline["r2"] else float("nan")
            ),
        }
        for model_name, values in (
            ("tec_smooth_baseline", baseline),
            ("xgboost_gim_prediction", model_metrics),
        ):
            row: dict[str, object] = {
                "split": split_name,
                "model": model_name,
                "n_samples": int(len(split_index)),
                **values,
            }
            if model_name == "xgboost_gim_prediction":
                row.update(improvements)
            rows.append(row)
        summary[split_name] = {
            "n_samples": int(len(split_index)),
            "tec_smooth_baseline": baseline,
            "xgboost_gim_prediction": model_metrics,
            "improvement": improvements,
        }

    summary["target_definition_check"] = {
        "formula": "target = gim_vtec",
        "maximum_absolute_mismatch": maximum_definition_error,
        "passed": True,
    }
    return pd.DataFrame(rows), summary


def save_gim_prediction_outputs(
    metrics_frame: pd.DataFrame,
    summary: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    """保存 GIM 预测指标并在日志中突出相对 TEC_smooth 基线的改善。"""
    metrics_frame.to_csv(output_dir / "gim_prediction_metrics.csv", index=False)
    available_splits = [
        name for name in ("train", "validation", "test") if name in summary
    ]
    comparison_rows = []
    for split_name in available_splits:
        split_summary = summary[split_name]
        baseline_values = split_summary["tec_smooth_baseline"]
        model_values = split_summary["xgboost_gim_prediction"]
        improvement_values = split_summary["improvement"]
        comparison_rows.append({
            "split": split_name,
            "n_samples": split_summary["n_samples"],
            **{
                f"baseline_{metric}": baseline_values[metric]
                for metric in ("rmse", "mae", "pearson", "r2")
            },
            **{
                f"model_{metric}": model_values[metric]
                for metric in ("rmse", "mae", "pearson", "r2")
            },
            **improvement_values,
        })
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "gim_prediction_comparison.csv", index=False
    )
    with (output_dir / "gim_prediction_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if "test" not in summary:
        raise RuntimeError("GIM prediction evaluation must include a test split")
    test = summary["test"]
    baseline = test["tec_smooth_baseline"]
    model_metrics = test["xgboost_gim_prediction"]
    improvement = test["improvement"]
    log(
        "Test GIM baseline (TEC_smooth -> GIM): "
        f"RMSE={baseline['rmse']:.6f}, MAE={baseline['mae']:.6f}, "
        f"R={baseline['pearson']:.6f}, R2={baseline['r2']:.6f}"
    )
    log(
        "Test direct GIM prediction: "
        f"RMSE={model_metrics['rmse']:.6f}, MAE={model_metrics['mae']:.6f}, "
        f"R={model_metrics['pearson']:.6f}, R2={model_metrics['r2']:.6f}; "
        f"RMSE improvement={improvement['rmse_improvement_percent']:.2f}%, "
        f"MAE improvement={improvement['mae_improvement_percent']:.2f}%"
    )


def build_tail_diagnostics(
    split_indices: dict[str, np.ndarray], y_values: np.ndarray,
    predictions: dict[str, np.ndarray], quantiles: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """按训练分位数评估五段，并计算每个集合的等权宏平均。"""
    rows: list[dict[str, object]] = []
    labels_by_split: dict[str, np.ndarray] = {}
    macro: dict[str, dict[str, float]] = {}
    metric_names = ("rmse", "mae", "bias")
    for split_name, split_index in split_indices.items():
        true = y_values[split_index]
        pred = predictions[split_name]
        labels = tail_segment_labels(true, quantiles)
        labels_by_split[split_name] = labels
        segment_metrics: list[dict[str, float]] = []
        if sum(int(np.sum(labels == value)) for value in TAIL_SEGMENTS) != len(true):
            raise RuntimeError(f"Tail segments do not cover the {split_name} split")
        for segment in TAIL_SEGMENTS:
            mask = labels == segment
            count = int(mask.sum())
            row: dict[str, object] = {
                "split": split_name,
                "segment": segment,
                "n_samples": count,
                "sample_ratio": count / len(true),
            }
            if count:
                values = regression_metrics(true[mask], pred[mask])
                row.update(values)
                segment_metrics.append(values)
            rows.append(row)
        macro[split_name] = {
            f"macro_{metric}": float(np.mean([item[metric] for item in segment_metrics]))
            for metric in metric_names
        }
    return pd.DataFrame(rows), labels_by_split, macro


def save_tail_metric_figure(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    """保存验证集和测试集的五段 RMSE/MAE/Bias 对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(TAIL_SEGMENTS))
    width = 0.36
    for offset, split_name in enumerate(("validation", "test")):
        subset = diagnostics.loc[diagnostics["split"] == split_name].set_index("segment")
        for axis, metric in zip(axes, ("rmse", "mae", "bias")):
            axis.bar(
                x + (offset - 0.5) * width,
                subset.reindex(TAIL_SEGMENTS)[metric],
                width,
                label=split_name,
            )
    for axis, title in zip(axes, ("RMSE", "MAE", "Bias (pred - true)")):
        axis.set_xticks(x, ("<q01", "q01-q05", "q05-q95", "q95-q99", ">q99"), rotation=20)
        axis.set_title(title)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        axis.legend()
    fig.suptitle("Metrics by training-derived target quantiles")
    fig.tight_layout()
    fig.savefig(output_dir / "tail_segment_metrics.png", dpi=200)
    plt.close(fig)


def save_prediction_files(
    data: pd.DataFrame, split_indices: dict[str, np.ndarray], y_values: np.ndarray,
    predictions: dict[str, np.ndarray], output_dir: Path,
) -> None:
    """为三个集合保存可按原始行 ID 回连的预测明细。"""
    frames = []
    for split_name, split_index in split_indices.items():
        true = y_values[split_index]
        pred = predictions[split_name]
        tec_smooth = data.iloc[split_index]["TEC_smooth"].to_numpy(dtype=np.float64)
        observed_residual = data.iloc[split_index][FILTER_COLUMN].to_numpy(dtype=np.float64)
        frames.append(pd.DataFrame({
            "original_row_index": data.iloc[split_index][ORIGINAL_ROW_COLUMN].to_numpy(),
            "source_file_id": data.iloc[split_index][SOURCE_FILE_COLUMN].to_numpy(),
            "source_row_number": data.iloc[split_index][SOURCE_ROW_COLUMN].to_numpy(),
            "split": split_name,
            "true_gim_vtec": true,
            "predicted_gim_vtec": pred,
            "prediction_error": pred - true,
            "absolute_error": np.abs(pred - true),
            "TEC_smooth_baseline": tec_smooth,
            "baseline_error": tec_smooth - true,
            "observed_residual": observed_residual,
            "predicted_gim_minus_TEC_smooth": pred - tec_smooth,
        }))
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "all_split_predictions.csv", index=False
    )


def sigma_segment_labels(z_score: np.ndarray) -> np.ndarray:
    """按绝对训练 z-score 生成四个互斥的 sigma 分段标签。"""
    absolute = np.abs(z_score)
    return np.select(
        [absolute <= 1, absolute <= 2, absolute <= 3],
        SIGMA_SEGMENTS[:3], default=SIGMA_SEGMENTS[3],
    )


def _distribution_statistics(prefix: str, values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    result = {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }
    for label, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), quantiles):
        result[f"{prefix}_{label}"] = float(value)
    return result


def build_sigma_diagnostics(
    raw_indices: dict[str, np.ndarray], y_values: np.ndarray,
    raw_predictions: dict[str, np.ndarray], target_mean: float, target_std: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """生成 train/validation/test 在互斥 sigma 区间上的长表诊断。"""
    rows: list[dict[str, object]] = []
    labels_by_split: dict[str, np.ndarray] = {}
    z_scores_by_split: dict[str, np.ndarray] = {}
    for split_name, split_index in raw_indices.items():
        true = y_values[split_index]
        predicted = raw_predictions[split_name]
        z_score = (true - target_mean) / target_std
        labels = sigma_segment_labels(z_score)
        labels_by_split[split_name] = labels
        z_scores_by_split[split_name] = z_score
        for segment in SIGMA_SEGMENTS:
            mask = labels == segment
            count = int(mask.sum())
            row: dict[str, object] = {
                "split": split_name,
                "segment": segment,
                "n_samples": count,
                "sample_ratio": count / len(split_index),
            }
            if count:
                true_segment = true[mask]
                pred_segment = predicted[mask]
                row.update(regression_metrics(true_segment, pred_segment))
                row.update(_distribution_statistics("true", true_segment))
                row.update(_distribution_statistics("pred", pred_segment))
                true_std = float(np.std(true_segment))
                pred_std = float(np.std(pred_segment))
                row["compression_ratio"] = (
                    pred_std / true_std if not np.isclose(true_std, 0.0) else float("nan")
                )
                row["fit_slope"] = (
                    float(np.polyfit(true_segment, pred_segment, 1)[0])
                    if count >= 2 and not np.isclose(true_std, 0.0)
                    else float("nan")
                )
            else:
                for column in (
                    "rmse", "mae", "r2", "mean_error", "error_std",
                    "true_min", "true_max", "true_mean", "true_std",
                    "true_p01", "true_p05", "true_p25", "true_p50", "true_p75",
                    "true_p95", "true_p99", "pred_min", "pred_max", "pred_mean",
                    "pred_std", "pred_p01", "pred_p05", "pred_p25", "pred_p50",
                    "pred_p75", "pred_p95", "pred_p99", "compression_ratio", "fit_slope",
                ):
                    row[column] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows), labels_by_split, z_scores_by_split


def save_sigma_diagnostic_figures(
    diagnostics: pd.DataFrame, raw_indices: dict[str, np.ndarray],
    y_values: np.ndarray, raw_predictions: dict[str, np.ndarray],
    labels_by_split: dict[str, np.ndarray], z_scores_by_split: dict[str, np.ndarray],
    output_dir: Path, maximum_points: int, seed: int,
) -> None:
    """保存 sigma 分段误差、散点、z-score误差和目标分布诊断图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(SIGMA_SEGMENTS))
    width = 0.25
    for offset, split_name in enumerate(("train", "validation", "test")):
        subset = diagnostics.loc[diagnostics["split"] == split_name].set_index("segment")
        for axis, metric in zip(axes, ("rmse", "mae")):
            axis.bar(x + (offset - 1) * width, subset.loc[list(SIGMA_SEGMENTS), metric], width, label=split_name)
    for axis, metric in zip(axes, ("RMSE", "MAE")):
        axis.set_xticks(x, ("<=1σ", "1–2σ", "2–3σ", ">3σ"))
        axis.set_ylabel(metric)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend()
    fig.suptitle("Error by mutually exclusive training-derived sigma segment")
    fig.tight_layout()
    fig.savefig(output_dir / "sigma_segment_metrics.png", dpi=200)
    plt.close(fig)

    for split_number, split_name in enumerate(("train", "validation", "test")):
        split_index = raw_indices[split_name]
        true = y_values[split_index]
        predicted = raw_predictions[split_name]
        labels = labels_by_split[split_name]
        limits = [float(min(true.min(), predicted.min())), float(max(true.max(), predicted.max()))]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
        for segment_number, (axis, segment) in enumerate(zip(axes.ravel(), SIGMA_SEGMENTS)):
            positions = np.flatnonzero(labels == segment)
            chosen = select_plot_indices(
                len(positions), maximum_points // 4 if maximum_points > 0 else 0,
                seed + split_number * 10 + segment_number,
            )
            positions = positions[chosen]
            axis.scatter(true[positions], predicted[positions], s=8, alpha=0.25)
            axis.plot(limits, limits, "r--", lw=1.2)
            axis.set_title(f"{segment} (n={int((labels == segment).sum()):,})")
            axis.grid(True, linestyle="--", alpha=0.3)
        fig.supxlabel("True GIM VTEC")
        fig.supylabel("Predicted GIM VTEC")
        fig.suptitle(f"{split_name}: true vs predicted by sigma segment")
        fig.tight_layout()
        fig.savefig(output_dir / f"{split_name}_sigma_segment_scatter.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for split_number, (axis, split_name) in enumerate(zip(axes, ("train", "validation", "test"))):
        true = y_values[raw_indices[split_name]]
        error = np.abs(raw_predictions[split_name] - true)
        chosen = select_plot_indices(len(true), maximum_points, seed + 100 + split_number)
        axis.scatter(z_scores_by_split[split_name][chosen], error[chosen], s=6, alpha=0.2)
        axis.axvline(-3, color="red", linestyle="--", lw=1)
        axis.axvline(3, color="red", linestyle="--", lw=1)
        axis.set_title(split_name)
        axis.set_xlabel("GIM VTEC z-score (training statistics)")
        axis.grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Absolute prediction error")
    fig.tight_layout()
    fig.savefig(output_dir / "z_score_vs_absolute_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for split_name in ("train", "validation", "test"):
        ax.hist(
            y_values[raw_indices[split_name]], bins=80, density=True,
            histtype="step", linewidth=1.5, label=split_name,
        )
    ax.set_xlabel("True GIM VTEC")
    ax.set_ylabel("Density")
    ax.set_title("Target distribution by split")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution_by_split.png", dpi=200)
    plt.close(fig)

    test_true = y_values[raw_indices["test"]]
    test_predicted = raw_predictions["test"]
    shared_range = (
        float(min(test_true.min(), test_predicted.min())),
        float(max(test_true.max(), test_predicted.max())),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        test_true, bins=80, range=shared_range, density=True,
        histtype="step", linewidth=1.6, label="true GIM VTEC",
    )
    ax.hist(
        test_predicted, bins=80, range=shared_range, density=True,
        histtype="step", linewidth=1.6, label="predicted GIM VTEC",
    )
    ax.set_xlabel("GIM VTEC")
    ax.set_ylabel("Density")
    ax.set_title("Full test: true and predicted GIM VTEC distributions")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "test_true_predicted_distribution.png", dpi=200)
    plt.close(fig)


def export_extreme_rows(
    data: pd.DataFrame, files: list[Path], raw_indices: dict[str, np.ndarray],
    y_values: np.ndarray, raw_predictions: dict[str, np.ndarray],
    labels_by_split: dict[str, np.ndarray], z_scores_by_split: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    """导出各 split 中超出训练 3σ 的完整原始记录及预测诊断字段。"""
    for split_name, split_index in raw_indices.items():
        mask = labels_by_split[split_name] == "beyond_3sigma"
        positions = np.flatnonzero(mask)
        frame = data.iloc[split_index[positions]].copy()
        frame.insert(0, "original_row_index", frame.pop(ORIGINAL_ROW_COLUMN).to_numpy())
        frame = frame.drop(columns=["index"], errors="ignore")
        source_ids = frame[SOURCE_FILE_COLUMN].to_numpy(dtype=int)
        frame.insert(1, "source_file", [str(files[value]) for value in source_ids])
        true = y_values[split_index][positions]
        predicted = raw_predictions[split_name][positions]
        frame["true_gim_vtec"] = true
        frame["predicted_gim_vtec"] = predicted
        frame["prediction_error"] = predicted - true
        frame["absolute_error"] = np.abs(predicted - true)
        frame["target_z_score"] = z_scores_by_split[split_name][positions]
        frame["sigma_segment"] = labels_by_split[split_name][positions]
        frame["tail_direction"] = np.where(frame["target_z_score"] > 3, "positive", "negative")
        output_name = f"{split_name}_extreme_beyond_3sigma.csv"
        frame.to_csv(output_dir / output_name, index=False)
        log(f"Exported {len(frame):,} extreme {split_name} rows to {output_name}")


def save_training_history(
    model: xgb.XGBRegressor, output_dir: Path
) -> dict[str, list[float]]:
    """提取模型训练历史，保存为CSV文件并绘制损失曲线图表。"""
    raw = model.evals_result()
    history = {
        "train_rmse": [float(value) for value in raw["validation_0"]["rmse"]],
        "validation_rmse": [float(value) for value in raw["validation_1"]["rmse"]],
    }
    pd.DataFrame({
        "iteration": np.arange(1, len(history["train_rmse"]) + 1), **history,
    }).to_csv(output_dir / "training_loss_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, len(history["train_rmse"]) + 1)
    ax.plot(iterations, history["train_rmse"], label="Training RMSE", lw=1.5)
    ax.plot(iterations, history["validation_rmse"], label="Validation RMSE", lw=1.5)
    best_iteration = int(np.argmin(history["validation_rmse"])) + 1
    ax.axvline(
        best_iteration, color="red", linestyle="--", alpha=0.8,
        label=f"Best iteration: {best_iteration}",
    )
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("RMSE loss")
    ax.set_title("XGBoost training loss by iteration")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curve.png", dpi=200)
    plt.close(fig)
    return history


def select_plot_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    """从数据中随机选择用于绘图的点的索引，避免绘图点过多。"""
    if maximum <= 0 or length <= maximum:
        return np.arange(length)
    return np.sort(np.random.default_rng(seed).choice(length, maximum, replace=False))


def save_test_figure(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path,
    maximum_points: int, seed: int,
) -> None:
    """绘制测试集真实-预测、误差-真实及误差分布诊断图。"""
    plot_idx = select_plot_indices(len(y_true), maximum_points, seed)
    errors = y_pred - y_true
    metrics = regression_metrics(y_true, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    density = axes[0].hexbin(
        y_true[plot_idx], y_pred[plot_idx], gridsize=80, mincnt=1, bins="log"
    )
    fig.colorbar(density, ax=axes[0], label="log10(count)")
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(bounds, bounds, "r--", lw=1.5)
    axes[0].set_xlabel("True GIM VTEC")
    axes[0].set_ylabel("Predicted GIM VTEC")
    axes[0].set_title(
        f"Test prediction (RMSE={metrics['rmse']:.5f}, R2={metrics['r2']:.4f})"
    )
    error_density = axes[1].hexbin(
        y_true[plot_idx], errors[plot_idx], gridsize=80, mincnt=1, bins="log"
    )
    fig.colorbar(error_density, ax=axes[1], label="log10(count)")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("True GIM VTEC")
    axes[1].set_ylabel("Prediction error (predicted - true)")
    axes[1].set_title("Error vs true GIM VTEC")
    axes[2].hist(errors, bins=50, color="#2ca02c", alpha=0.75)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_xlabel("Prediction error (predicted - true)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Test error distribution")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_importance(
    model: xgb.XGBRegressor, feature_columns: list[str], output_dir: Path
) -> None:
    """计算并保存特征重要性，输出完整排序表和Top15特征的柱状图。"""
    importance = pd.DataFrame({
        "feature": feature_columns, "gain_importance": model.feature_importances_,
    }).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    top = importance.head(15).sort_values("gain_importance")
    ax.barh(top["feature"], top["gain_importance"], color="#ff7f0e")
    ax.set_xlabel("Feature importance")
    ax.set_title("Top feature importances")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top15.png", dpi=200)
    plt.close(fig)


def save_eda_figures(data: pd.DataFrame, output_dir: Path) -> None:
    """生成并保存探索性数据分析图表，包括直方图和相关性矩阵。"""
    numeric_data = data.select_dtypes(include=[np.number]).drop(
        columns=[SOURCE_FILE_COLUMN, SOURCE_ROW_COLUMN, ORIGINAL_ROW_COLUMN], errors="ignore"
    )
    axes = numeric_data.hist(figsize=(16, 12), bins=40)
    figure = axes.ravel()[0].figure
    figure.tight_layout()
    figure.savefig(output_dir / "eda_histograms.png", dpi=180)
    plt.close(figure)

    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(image)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.tight_layout()
    fig.savefig(output_dir / "eda_correlation.png", dpi=180)
    plt.close(fig)


def evaluate_existing_run(output_dir: Path) -> None:
    """从新流程已有预测中重算 GIM 直接预测指标，不重新训练。"""
    all_predictions_path = output_dir / "all_split_predictions.csv"
    test_predictions_path = output_dir / "test_predictions.csv"
    predictions_path = (
        all_predictions_path
        if all_predictions_path.is_file()
        else test_predictions_path
    )
    manifest_path = output_dir / "input_file_manifest.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            "Existing predictions not found; expected either "
            f"{all_predictions_path} or {test_predictions_path}"
        )
    predictions_frame = pd.read_csv(predictions_path).reset_index(drop=True)
    if "split" not in predictions_frame.columns:
        if predictions_path == test_predictions_path:
            predictions_frame["split"] = "test"
        else:
            raise ValueError(f"{predictions_path.name} is missing column: split")
    required_prediction_columns = {
        "source_file_id", "source_row_number", "split",
        "true_gim_vtec", "predicted_gim_vtec",
    }
    missing = required_prediction_columns.difference(predictions_frame.columns)
    if missing:
        raise ValueError(
            f"{predictions_path.name} is missing columns: {sorted(missing)}"
        )

    if "TEC_smooth_baseline" in predictions_frame.columns:
        tec_smooth = predictions_frame["TEC_smooth_baseline"].to_numpy(
            dtype=np.float64
        )
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path).set_index("source_file_id")
        tec_smooth = np.full(len(predictions_frame), np.nan, dtype=np.float64)
        for source_file_id, positions in predictions_frame.groupby(
            "source_file_id", sort=False
        ).groups.items():
            source_id = int(source_file_id)
            if source_id not in manifest.index:
                raise KeyError(f"source_file_id {source_id} is absent from the manifest")
            source_path = Path(str(manifest.loc[source_id, "path"]))
            if not source_path.is_file():
                raise FileNotFoundError(f"Source CSV no longer exists: {source_path}")
            source = pd.read_csv(source_path, usecols=["TEC_smooth"])
            position_array = np.asarray(positions, dtype=np.int64)
            row_numbers = predictions_frame.loc[
                position_array, "source_row_number"
            ].to_numpy(dtype=np.int64)
            if np.any(row_numbers < 0) or np.any(row_numbers >= len(source)):
                raise IndexError(f"Source-row index is outside {source_path}")
            tec_smooth[position_array] = source.iloc[row_numbers][
                "TEC_smooth"
            ].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(tec_smooth)):
            raise RuntimeError("Failed to recover finite TEC_smooth values for every prediction")

    target_values = predictions_frame["true_gim_vtec"].to_numpy(dtype=np.float64)
    data = pd.DataFrame({"TEC_smooth": tec_smooth, TARGET_COLUMN: target_values})
    split_values = predictions_frame["split"].astype(str).to_numpy()
    split_indices = {
        name: np.flatnonzero(split_values == name)
        for name in ("train", "validation", "test")
        if np.any(split_values == name)
    }
    if "test" not in split_indices:
        raise ValueError("Existing predictions must contain a test split")
    target_predictions = {
        name: predictions_frame.iloc[index]["predicted_gim_vtec"].to_numpy(
            dtype=np.float64
        )
        for name, index in split_indices.items()
    }
    metrics_frame, summary = evaluate_gim_prediction(
        data, split_indices, target_values, target_predictions
    )
    save_gim_prediction_outputs(metrics_frame, summary, output_dir)

    predictions_frame["TEC_smooth_baseline"] = tec_smooth
    predictions_frame["baseline_error"] = tec_smooth - target_values
    predictions_frame["predicted_gim_minus_TEC_smooth"] = (
        predictions_frame["predicted_gim_vtec"].to_numpy(dtype=np.float64)
        - tec_smooth
    )
    augmented_name = (
        "all_split_predictions_recalculated.csv"
        if predictions_path == all_predictions_path
        else "test_predictions_recalculated.csv"
    )
    predictions_frame.to_csv(output_dir / augmented_name, index=False)
    predictions_frame.loc[predictions_frame["split"] == "test"].to_csv(
        output_dir / "test_gim_predictions_recalculated.csv", index=False
    )

    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            run_metrics = json.load(handle)
        run_metrics["gim_prediction_evaluation"] = summary
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(run_metrics, handle, ensure_ascii=False, indent=2)
    log(f"Evaluation-only outputs written to {output_dir.resolve()}")


def main() -> int:
    """主函数：完整执行数据处理、模型优化、训练、评估和结果保存的整个流程。"""
    args = parse_args()
    validate_args(args)
    if args.evaluation_only:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(
                f"Existing result directory not found: {args.output_dir}"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    log("Command line: " + shlex.join(sys.argv))
    if args.evaluation_only:
        evaluate_existing_run(args.output_dir)
        return 0

    assert args.input_dir is not None
    files = find_csv_files(args.input_dir, args.pattern, args.recursive)
    log(f"Found {len(files):,} CSV files")
    pd.DataFrame([
        {
            "source_file_id": number,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "modified_time": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        }
        for number, path in enumerate(files)
    ]).to_csv(args.output_dir / "input_file_manifest.csv", index=False)
    data = load_csv_files(files, args.csv_engine, args.read_batch_size)
    (
        data, feature_columns, fixed_altitude, original_altitude,
        qc_log, qc_removed_records,
    ) = clean_and_validate(data, args.fixed_altitude)
    pd.DataFrame([qc_log]).to_csv(args.output_dir / "qc_summary.csv", index=False)
    qc_removed_records.to_csv(args.output_dir / "qc_removed_records.csv", index=False)
    data, sampling_summary, daily_sampling = sample_rows_within_days(
        data, args.sample_fraction, args.sample_seed
    )
    log(f"Model features ({len(feature_columns)}): {feature_columns}")
    if args.save_eda:
        save_eda_figures(data, args.output_dir)

    forbidden_features = {
        TARGET_COLUMN, FILTER_COLUMN, "is_3sigma", "is_iqr_outlier",
        "tail_segment", "sample_weight",
    }
    leaked_features = sorted(forbidden_features.intersection(feature_columns))
    if leaked_features:
        raise RuntimeError(f"Target-derived columns cannot be model features: {leaked_features}")

    # 先按时间排序以保证输出稳定，再随机分配不可拆分的完整自然日。
    data_sorted = data.sort_values("datetime").reset_index(drop=False)
    original_indices = data_sorted[ORIGINAL_ROW_COLUMN].to_numpy()
    data_sorted = data_sorted.reset_index(drop=True)
    indices, _day_assignments = split_by_calendar_day(
        data_sorted, daily_sampling, args.split_strategy, args.split_seed,
        args.train_ratio, args.validation_ratio, args.test_ratio,
        args.output_dir,
    )
    x_values = data_sorted[feature_columns].to_numpy(dtype=np.float32)
    y_values = data_sorted[TARGET_COLUMN].to_numpy(dtype=np.float32)
    if np.any(data_sorted[FILTER_COLUMN].to_numpy(dtype=np.float64) <= 0):
        raise RuntimeError("Hard QC failed: residual <= 0 remains after cleaning")
    split_summary_rows = []
    for split_name, split_index in indices.items():
        split_frame = data_sorted.iloc[split_index]
        row: dict[str, object] = {
            "split": split_name,
            "rows": int(len(split_index)),
            "calendar_days": int(split_frame["datetime"].dt.normalize().nunique()),
            "start_time": split_frame["datetime"].min().isoformat(),
            "end_time": split_frame["datetime"].max().isoformat(),
            "latitude_min": float(split_frame["lat"].min()),
            "latitude_max": float(split_frame["lat"].max()),
            "longitude_min": float(split_frame["lon"].min()),
            "longitude_max": float(split_frame["lon"].max()),
        }
        row.update(_distribution_statistics(TARGET_COLUMN, y_values[split_index]))
        split_summary_rows.append(row)
    pd.DataFrame(split_summary_rows).to_csv(
        args.output_dir / "split_summary.csv", index=False
    )
    x_train, y_train = x_values[indices["train"]], y_values[indices["train"]]
    x_validation, y_validation = (
        x_values[indices["validation"]], y_values[indices["validation"]]
    )

    study = optimize_parameters(
        args, x_train, y_train, x_validation, y_validation, None
    )
    study.trials_dataframe().to_csv(args.output_dir / "optuna_trials.csv", index=False)
    with (args.output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(study.best_params, handle, ensure_ascii=False, indent=2)

    final_params = {**base_parameters(args), **study.best_params}
    log(
        "Training selected model with "
        f"n_estimators={args.n_estimators}, reg_alpha={args.reg_alpha}, "
        f"reg_lambda={args.reg_lambda}, tuned parameters={study.best_params}"
    )
    model = fit_with_early_stopping(
        final_params, x_train, y_train, x_validation, y_validation,
        args.early_stopping_rounds, None,
    )
    history = save_training_history(model, args.output_dir)
    model.save_model(args.output_dir / "xg_jason_gim_vtec_model.json")
    save_feature_importance(model, feature_columns, args.output_dir)

    evaluations: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    evaluation_indices = {
        "train": indices["train"],
        "validation": indices["validation"],
        "test": indices["test"],
    }
    for name, split_index in evaluation_indices.items():
        prediction = model.predict(x_values[split_index])
        predictions[name] = prediction
        evaluations[name] = regression_metrics(y_values[split_index], prediction)
        log(
            f"{name}: RMSE={evaluations[name]['rmse']:.6f}, "
            f"MAE={evaluations[name]['mae']:.6f}, R2={evaluations[name]['r2']:.6f}"
        )

    predictions = {
        name: model.predict(x_values[split_index])
        for name, split_index in indices.items()
    }
    gim_metrics_frame, gim_prediction_summary = evaluate_gim_prediction(
        data_sorted, indices, y_values, predictions
    )
    save_gim_prediction_outputs(
        gim_metrics_frame, gim_prediction_summary, args.output_dir
    )
    save_prediction_files(
        data_sorted, indices, y_values, predictions, args.output_dir,
    )

    test_index = indices["test"]
    test_predictions_frame = pd.DataFrame({
        "original_row_index": original_indices[test_index],
        "source_file_id": data_sorted.iloc[test_index][SOURCE_FILE_COLUMN].to_numpy(),
        "source_row_number": data_sorted.iloc[test_index][SOURCE_ROW_COLUMN].to_numpy(),
        "true_gim_vtec": y_values[test_index],
        "predicted_gim_vtec": predictions["test"],
        "prediction_error": predictions["test"] - y_values[test_index],
        "absolute_error": np.abs(predictions["test"] - y_values[test_index]),
        "TEC_smooth_baseline": data_sorted.iloc[test_index]["TEC_smooth"].to_numpy(),
        "baseline_error": (
            data_sorted.iloc[test_index]["TEC_smooth"].to_numpy()
            - y_values[test_index]
        ),
        "observed_residual": data_sorted.iloc[test_index][FILTER_COLUMN].to_numpy(),
        "predicted_gim_minus_TEC_smooth": (
            predictions["test"]
            - data_sorted.iloc[test_index]["TEC_smooth"].to_numpy()
        ),
    })
    test_predictions_frame.to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    save_test_figure(
        y_values[test_index], predictions["test"],
        args.output_dir / "test_performance.png",
        args.max_scatter_points, args.model_seed,
    )

    metrics = {
        "input_directory": str(args.input_dir.resolve()),
        "input_file_count": len(files),
        "input_pattern": args.pattern,
        "input_recursive": args.recursive,
        "command_line": shlex.join(sys.argv),
        "valid_row_count_before_sampling": sampling_summary["rows_before_sampling"],
        "analyzed_row_count": len(data),
        "sampling": sampling_summary,
        "hard_qc": qc_log,
        "split_method": f"{args.split_strategy} complete-calendar-day split",
        "split_ratio_target": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "split_rule": {
            "grouping_unit": "calendar date",
            "assignment_seed": args.split_seed,
            "strategy": args.split_strategy,
            "same_day_cross_split_allowed": False,
            "day_assignment_file": "day_assignments.csv",
            "row_assignment_file": "split_assignments.csv",
        },
        "model_split_rows": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "test": len(indices["test"]),
        },
        "split_row_ratios": {
            name: len(value) / len(data_sorted) for name, value in indices.items()
        },
        "split_calendar_days": {
            name: int(data_sorted.iloc[index]["datetime"].dt.normalize().nunique())
            for name, index in indices.items()
        },
        "date_coverage": {
            name: {
                "start": str(
                    data_sorted.iloc[index]["datetime"].dt.normalize().min().date()
                ),
                "end": str(
                    data_sorted.iloc[index]["datetime"].dt.normalize().max().date()
                ),
            }
            for name, index in indices.items()
        },
        "model_seed": args.model_seed,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "target_column": TARGET_COLUMN,
        "row_filter": "residual > 0 retained before sampling and splitting",
        "regularization": {
            "l1_reg_alpha": args.reg_alpha,
            "l2_reg_lambda": args.reg_lambda,
        },
        "maximum_boosting_rounds": args.n_estimators,
        "gim_prediction_evaluation": gim_prediction_summary,
        "time_features": {
            "source_datetime_timezone": "UTC",
            "local_solar_hour_formula": "(UTC fractional hour + longitude / 15) modulo 24",
            "columns": [LOCAL_TIME_SIN_COLUMN, LOCAL_TIME_COS_COLUMN],
            "replaced_columns": ["HOD_s", "HOD_c"],
        },
        "feature_columns": feature_columns,
        "fixed_altitude": fixed_altitude,
        "original_altitude": original_altitude,
        "optuna": {
            "completed_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_validation_rmse": study.best_value,
            "best_parameters": study.best_params,
        },
        "selected_model_best_iteration": int(
            getattr(model, "best_iteration", len(history["train_rmse"]) - 1)
        ),
        "evaluations": evaluations,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    log(f"Finished. Results written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr, flush=True)
        raise SystemExit(130)
