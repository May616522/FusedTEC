#!/usr/bin/env python3
"""Tune and train the Jason ionospheric-residual XGBoost model.

Rows are sampled independently within every calendar day so that all available
days remain represented. Complete days are then randomly assigned to
train/validation/test; one day can belong to only one split. Local solar-time
sine/cosine features are calculated from UTC timestamps and longitude, replacing
the original UTC-hour features. Sigma thresholds always come from the raw
training split and are reused for all diagnostics.
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
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "residual"
SOURCE_FILE_COLUMN = "__source_file_id"
SOURCE_ROW_COLUMN = "__source_row_number"
ORIGINAL_ROW_COLUMN = "__cleaned_row_index"
LOCAL_TIME_SIN_COLUMN = "local_time_s"
LOCAL_TIME_COS_COLUMN = "local_time_c"
EXCLUDED_COLUMNS = {
    "datetime", TARGET_COLUMN, "TEC_smooth", "TEC_raw", SOURCE_FILE_COLUMN,
    SOURCE_ROW_COLUMN, ORIGINAL_ROW_COLUMN, "HOD_s", "HOD_c",
}
SIGMA_SEGMENTS = ("within_1sigma", "sigma_1_to_2", "sigma_2_to_3", "beyond_3sigma")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括输入/输出目录、Optuna试验数量、XGBoost参数范围等。"""
    parser = argparse.ArgumentParser(
        description=(
            "Optuna-tuned XGBoost regression for Jason residuals with a "
            "random, calendar-day-grouped train/validation/test split."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--csv-engine", choices=("c", "python", "pyarrow"), default="c")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--optuna-timeout", type=int, default=3600,
        help="Optuna time limit in seconds; 0 disables the limit.",
    )
    parser.add_argument("--min-estimators", type=int, default=200)
    parser.add_argument("--max-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument(
        "--sample-fraction", type=float, default=1.0,
        help=(
            "Randomly retain this fraction within every valid calendar day. "
            "Every day keeps at least one row; default: 1.0."
        ),
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed used only for --sample-fraction; default: 42.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for assigning complete days to splits; default: 42.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--fixed-altitude", type=float, default=None,
        help="Replace alt by this constant; default is the cleaned-data median.",
    )
    parser.add_argument(
        "--outlier-sigma", type=float, default=3.0,
        help=(
            "Remove target rows outside mean +/- N population standard deviations "
            "using bounds calculated from the training split only. The test split "
            "is not filtered; default: 3.0."
        ),
    )
    sigma_group = parser.add_mutually_exclusive_group()
    sigma_group.add_argument(
        "--sigma-filter", dest="sigma_filter", action="store_true",
        help="Filter training and validation targets using training-derived bounds (default).",
    )
    sigma_group.add_argument(
        "--no-sigma-filter", dest="sigma_filter", action="store_false",
        help="Keep target-tail rows in training and validation; diagnostics are still produced.",
    )
    parser.set_defaults(sigma_filter=True)
    parser.add_argument(
        "--export-extremes", action="store_true",
        help="Export all train/validation/test rows beyond the training-derived 3-sigma bounds.",
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
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if args.optuna_timeout < 0:
        raise ValueError("--optuna-timeout cannot be negative")
    if args.min_estimators < 1 or args.max_estimators < args.min_estimators:
        raise ValueError("Estimator range is invalid")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds cannot be negative")
    if not np.isfinite(args.outlier_sigma) or args.outlier_sigma <= 0:
        raise ValueError("--outlier-sigma must be a positive finite number")
    if not np.isfinite(args.sample_fraction) or not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in the interval (0, 1]")
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
) -> tuple[pd.DataFrame, list[str], float, dict[str, float]]:
    """清理缺失/无穷值，验证数据完整性，并固定高度值。"""
    required = {TARGET_COLUMN, "datetime", "lat", "lon", "alt", "gim_vtec"}
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
    data = data.dropna(axis=0, how="any").reset_index(drop=True)
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
    data.loc[:, "alt"] = fixed_altitude
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
    return data, feature_columns, fixed_altitude, original_altitude


def build_sigma_filtered_splits(
    data: pd.DataFrame, splits: dict[str, np.ndarray], outlier_sigma: float,
    apply_filter: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """仅用训练集计算 residual 阈值，过滤训练/验证并标记测试 clean 子集。"""
    train_residual = data.iloc[splits["train"]][TARGET_COLUMN].to_numpy()
    residual_mean = float(np.mean(train_residual))
    residual_std = float(np.std(train_residual, ddof=0))
    if not np.isfinite(residual_mean) or not np.isfinite(residual_std) or residual_std <= 0:
        raise ValueError("Training residual mean/std is not finite or std is zero")

    lower_bound = residual_mean - outlier_sigma * residual_std
    upper_bound = residual_mean + outlier_sigma * residual_std
    residual = data[TARGET_COLUMN].to_numpy()
    clean_mask = (residual >= lower_bound) & (residual <= upper_bound)

    filtered_splits = {
        "train": (
            splits["train"][clean_mask[splits["train"]]]
            if apply_filter else splits["train"]
        ),
        "validation": (
            splits["validation"][clean_mask[splits["validation"]]]
            if apply_filter else splits["validation"]
        ),
        # 测试集保持完整，不参与 residual 筛选。
        "test": splits["test"],
    }
    test_clean_index = splits["test"][clean_mask[splits["test"]]]
    if len(filtered_splits["train"]) == 0 or len(filtered_splits["validation"]) == 0:
        raise ValueError("Sigma filtering produced an empty training or validation split")
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
            if name == "test" or not apply_filter
            else "within bounds and retained for modeling"
        )
        log(
            f"{name} sigma summary: {retained:,}/{total:,} within bounds; "
            f"{action}"
        )

    sigma_filter: dict[str, object] = {
        "fit_split": "train",
        "sigma_multiplier": float(outlier_sigma),
        "mean": residual_mean,
        "population_std": residual_std,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "diagnostic_bounds": {
            f"{level}_sigma": {
                "lower": float(residual_mean - level * residual_std),
                "upper": float(residual_mean + level * residual_std),
            }
            for level in (1, 2, 3)
        },
        "filter_enabled": bool(apply_filter),
        "application": {
            "train": "filtered" if apply_filter else "not filtered",
            "validation": "filtered" if apply_filter else "not filtered",
            "test": "not filtered; clean subset only marked for additional evaluation",
        },
        "splits": split_summary,
    }
    log(
        f"Training-only residual {outlier_sigma:g}-sigma bounds: "
        f"mean={residual_mean:.6f}, std={residual_std:.6f}, "
        f"bounds=[{lower_bound:.6f}, {upper_bound:.6f}]"
    )
    return filtered_splits, test_clean_index, sigma_filter


def split_by_calendar_day(
    data: pd.DataFrame, daily_sampling: pd.DataFrame, split_seed: int,
    train_ratio: float, validation_ratio: float, test_ratio: float,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """按完整自然日随机划分，并按月份分层以维持季节覆盖。"""
    date_frame = daily_sampling[[
        "calendar_date", "year", "day_of_year", "month",
        "rows_before_sampling", "rows_after_sampling", "realized_fraction",
    ]].copy()
    if len(date_frame) < 7:
        raise ValueError("Too few calendar days for a grouped train/validation/test split")
    temporary_ratio = validation_ratio + test_ratio
    try:
        train_days, temporary_days = train_test_split(
            date_frame,
            train_size=train_ratio,
            random_state=split_seed,
            shuffle=True,
            stratify=date_frame["month"],
        )
        validation_days, test_days = train_test_split(
            temporary_days,
            train_size=validation_ratio / temporary_ratio,
            random_state=split_seed + 1,
            shuffle=True,
            stratify=temporary_days["month"],
        )
        stratified_by_month = True
    except ValueError as error:
        log(f"Monthly stratification unavailable ({error}); using unstratified day split")
        train_days, temporary_days = train_test_split(
            date_frame, train_size=train_ratio, random_state=split_seed, shuffle=True
        )
        validation_days, test_days = train_test_split(
            temporary_days,
            train_size=validation_ratio / temporary_ratio,
            random_state=split_seed + 1,
            shuffle=True,
        )
        stratified_by_month = False

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
        "calendar_date": data["datetime"].dt.normalize(), "split": row_split,
    })
    if row_assignments.groupby("calendar_date")["split"].nunique().max() != 1:
        raise RuntimeError("At least one calendar date was assigned to multiple splits")
    day_assignments["calendar_date"] = day_assignments["calendar_date"].dt.strftime("%Y-%m-%d")
    day_assignments.to_csv(output_dir / "day_assignments.csv", index=False)
    monthly_summary = pd.crosstab(day_assignments["month"], day_assignments["split"])
    monthly_summary = monthly_summary.reindex(
        columns=["train", "validation", "test"], fill_value=0
    )
    monthly_summary.to_csv(output_dir / "monthly_day_split_summary.csv")

    log(
        "Dataset split: random complete-calendar-day assignment with ratios "
        f"{train_ratio:g}/{validation_ratio:g}/{test_ratio:g}; seed={split_seed}; "
        f"monthly stratification={'enabled' if stratified_by_month else 'fallback disabled'}"
    )
    for name, index in splits.items():
        split_dates = data.iloc[index]["datetime"].dt.normalize()
        log(
            f"{name}: rows={len(index):,} ({len(index) / total:.2%}), "
            f"calendar days={split_dates.nunique():,}, "
            f"date coverage={split_dates.min().date()} to {split_dates.max().date()}"
        )
    return splits, day_assignments


def base_parameters(args: argparse.Namespace) -> dict[str, object]:
    """生成XGBoost模型的基础参数配置。"""
    return {
        "objective": "reg:squarederror", "eval_metric": "rmse",
        "booster": "gbtree", "tree_method": "hist",
        "random_state": args.model_seed, "n_jobs": args.n_jobs, "verbosity": 0,
    }


def fit_with_early_stopping(
    params: dict[str, object], x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray, rounds: int,
) -> xgb.XGBRegressor:
    """使用早停策略训练XGBoost回归模型，兼容不同版本的XGBoost API。"""
    eval_set = [(x_train, y_train), (x_validation, y_validation)]
    if rounds == 0:
        model = xgb.XGBRegressor(**params)
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        return model
    model = xgb.XGBRegressor(**params)
    try:
        model.fit(
            x_train, y_train, eval_set=eval_set,
            early_stopping_rounds=rounds, verbose=False,
        )
    except TypeError:
        # XGBoost >= 2.1 moved early_stopping_rounds into the constructor.
        model = xgb.XGBRegressor(**params, early_stopping_rounds=rounds)
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    return model


def optimize_parameters(
    args: argparse.Namespace, x_train: np.ndarray, y_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray,
) -> optuna.Study:
    """使用Optuna框架进行XGBoost超参数调优，通过验证集RMSE最小化。"""
    fixed = base_parameters(args)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **fixed,
            "n_estimators": trial.suggest_int(
                "n_estimators", args.min_estimators, args.max_estimators
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
        }
        model = fit_with_early_stopping(
            params, x_train, y_train, x_validation, y_validation,
            args.early_stopping_rounds,
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
        study_name="xg_jason_ion_residual",
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
    """计算回归模型的评估指标（RMSE、MAE、R2、平均误差、误差标准差）。"""
    r2 = float("nan")
    if len(y_true) >= 2 and not np.isclose(np.var(y_true), 0.0):
        r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "mean_error": float(np.mean(y_pred - y_true)),
        "error_std": float(np.std(y_pred - y_true)),
    }


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
    raw_predictions: dict[str, np.ndarray], residual_mean: float, residual_std: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """生成 train/validation/test 在互斥 sigma 区间上的长表诊断。"""
    rows: list[dict[str, object]] = []
    labels_by_split: dict[str, np.ndarray] = {}
    z_scores_by_split: dict[str, np.ndarray] = {}
    for split_name, split_index in raw_indices.items():
        true = y_values[split_index]
        predicted = raw_predictions[split_name]
        z_score = (true - residual_mean) / residual_std
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
        fig.supxlabel("True residual")
        fig.supylabel("Predicted residual")
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
        axis.set_xlabel("Residual z-score (training statistics)")
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
    ax.set_xlabel("True residual")
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
        histtype="step", linewidth=1.6, label="true residual",
    )
    ax.hist(
        test_predicted, bins=80, range=shared_range, density=True,
        histtype="step", linewidth=1.6, label="predicted residual",
    )
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Full test: true and predicted residual distributions")
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
        frame["true_residual"] = true
        frame["predicted_residual"] = predicted
        frame["prediction_error"] = predicted - true
        frame["absolute_error"] = np.abs(predicted - true)
        frame["residual_z_score"] = z_scores_by_split[split_name][positions]
        frame["sigma_segment"] = labels_by_split[split_name][positions]
        frame["tail_direction"] = np.where(frame["residual_z_score"] > 3, "positive", "negative")
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
    """绘制测试集的预测性能图表，包括预测值vs真实值散点图和误差分布直方图。"""
    plot_idx = select_plot_indices(len(y_true), maximum_points, seed)
    errors = y_pred - y_true
    metrics = regression_metrics(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true[plot_idx], y_pred[plot_idx], s=8, alpha=0.2)
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(bounds, bounds, "r--", lw=1.5)
    axes[0].set_xlabel("True residual")
    axes[0].set_ylabel("Predicted residual")
    axes[0].set_title(
        f"Test prediction (RMSE={metrics['rmse']:.5f}, R2={metrics['r2']:.4f})"
    )
    axes[1].hist(errors, bins=50, color="#2ca02c", alpha=0.75)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Prediction error (predicted - true)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Test error distribution")
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


def main() -> int:
    """主函数：完整执行数据处理、模型优化、训练、评估和结果保存的整个流程。"""
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log("Command line: " + shlex.join(sys.argv))

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
    data, feature_columns, fixed_altitude, original_altitude = clean_and_validate(
        data, args.fixed_altitude
    )
    data, sampling_summary, daily_sampling = sample_rows_within_days(
        data, args.sample_fraction, args.sample_seed
    )
    log(f"Model features ({len(feature_columns)}): {feature_columns}")
    if args.save_eda:
        save_eda_figures(data, args.output_dir)

    # 先按时间排序以保持输出稳定，再把完整自然日随机分配到三个集合。
    data_sorted = data.sort_values("datetime").reset_index(drop=False)
    original_indices = data_sorted[ORIGINAL_ROW_COLUMN].to_numpy()
    data_sorted = data_sorted.reset_index(drop=True)
    raw_indices, day_assignments = split_by_calendar_day(
        data_sorted, daily_sampling, args.split_seed,
        args.train_ratio, args.validation_ratio, args.test_ratio,
        args.output_dir,
    )
    indices, test_clean_index, sigma_filter = build_sigma_filtered_splits(
        data_sorted, raw_indices, args.outlier_sigma, args.sigma_filter
    )
    x_values = data_sorted[feature_columns].to_numpy(dtype=np.float32)
    y_values = data_sorted[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_train, y_train = x_values[indices["train"]], y_values[indices["train"]]
    x_validation, y_validation = (
        x_values[indices["validation"]], y_values[indices["validation"]]
    )

    study = optimize_parameters(args, x_train, y_train, x_validation, y_validation)
    study.trials_dataframe().to_csv(args.output_dir / "optuna_trials.csv", index=False)
    with (args.output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(study.best_params, handle, ensure_ascii=False, indent=2)

    final_params = {**base_parameters(args), **study.best_params}
    log(f"Training selected model with parameters: {study.best_params}")
    model = fit_with_early_stopping(
        final_params, x_train, y_train, x_validation, y_validation,
        args.early_stopping_rounds,
    )
    history = save_training_history(model, args.output_dir)
    model.save_model(args.output_dir / "xg_jason_ion_model.json")
    save_feature_importance(model, feature_columns, args.output_dir)

    evaluations: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    evaluation_indices = {
        "train": indices["train"],
        "validation": indices["validation"],
        "test_full": indices["test"],
        "test_clean": test_clean_index,
    }
    for name, split_index in evaluation_indices.items():
        prediction = model.predict(x_values[split_index])
        predictions[name] = prediction
        evaluations[name] = regression_metrics(y_values[split_index], prediction)
        log(
            f"{name}: RMSE={evaluations[name]['rmse']:.6f}, "
            f"MAE={evaluations[name]['mae']:.6f}, R2={evaluations[name]['r2']:.6f}"
        )

    raw_predictions = {
        name: model.predict(x_values[split_index])
        for name, split_index in raw_indices.items()
    }
    sigma_diagnostics, labels_by_split, z_scores_by_split = build_sigma_diagnostics(
        raw_indices, y_values, raw_predictions,
        float(sigma_filter["mean"]), float(sigma_filter["population_std"]),
    )
    sigma_diagnostics.to_csv(args.output_dir / "sigma_segment_metrics.csv", index=False)
    for row in sigma_diagnostics.itertuples(index=False):
        log(
            f"sigma diagnostic {row.split}/{row.segment}: n={row.n_samples:,} "
            f"({row.sample_ratio:.2%}), RMSE={row.rmse:.6f}, "
            f"MAE={row.mae:.6f}, R2={row.r2:.6f}"
        )
    save_sigma_diagnostic_figures(
        sigma_diagnostics, raw_indices, y_values, raw_predictions,
        labels_by_split, z_scores_by_split, args.output_dir,
        args.max_scatter_points, args.model_seed,
    )
    if args.export_extremes:
        export_extreme_rows(
            data_sorted, files, raw_indices, y_values, raw_predictions,
            labels_by_split, z_scores_by_split, args.output_dir,
        )

    test_index = indices["test"]
    test_clean_mask = np.isin(test_index, test_clean_index)
    test_predictions_frame = pd.DataFrame({
        "original_row_index": original_indices[test_index],
        "source_file_id": data_sorted.iloc[test_index][SOURCE_FILE_COLUMN].to_numpy(),
        "source_row_number": data_sorted.iloc[test_index][SOURCE_ROW_COLUMN].to_numpy(),
        "true_residual": y_values[test_index],
        "predicted_residual": predictions["test_full"],
        "prediction_error": predictions["test_full"] - y_values[test_index],
        "absolute_error": np.abs(predictions["test_full"] - y_values[test_index]),
        "residual_z_score": z_scores_by_split["test"],
        "sigma_segment": labels_by_split["test"],
        "is_clean_by_training_sigma": test_clean_mask,
    })
    test_predictions_frame.to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    test_predictions_frame.loc[test_clean_mask].to_csv(
        args.output_dir / "test_clean_predictions.csv", index=False
    )
    save_test_figure(
        y_values[test_index], predictions["test_full"],
        args.output_dir / "test_performance.png",
        args.max_scatter_points, args.model_seed,
    )
    save_test_figure(
        y_values[test_clean_index], predictions["test_clean"],
        args.output_dir / "test_clean_performance.png",
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
        "residual_sigma_filter": sigma_filter,
        "split_method": "random complete-calendar-day split stratified by month",
        "split_ratio_target": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "split_rule": {
            "grouping_unit": "calendar date",
            "assignment_seed": args.split_seed,
            "stratification": "calendar month when feasible",
            "same_day_cross_split_allowed": False,
            "assignment_file": "day_assignments.csv",
        },
        "split_rows_before_sigma": {
            name: len(value) for name, value in raw_indices.items()
        },
        "model_split_rows": {
            "train": len(indices["train"]),
            "validation": len(indices["validation"]),
            "test_full": len(indices["test"]),
            "test_clean": len(test_clean_index),
        },
        "split_row_ratios": {
            name: len(value) / len(data_sorted) for name, value in raw_indices.items()
        },
        "split_calendar_days": {
            name: int(data_sorted.iloc[index]["datetime"].dt.normalize().nunique())
            for name, index in raw_indices.items()
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
            for name, index in raw_indices.items()
        },
        "model_seed": args.model_seed,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "sigma_filter_enabled": args.sigma_filter,
        "extreme_rows_exported": args.export_extremes,
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
