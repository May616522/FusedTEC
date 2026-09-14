#!/usr/bin/env python3
"""Tune and train the Jason ionospheric-residual XGBoost model.

The cleaned observations are split by temporal order into train/validation/test
sets using a 7:1:2 ratio. The training set comprises the earliest 70% of data,
validation set the middle 10%, and test set the most recent 20% (approximately
the last month). Optuna sees only the training and validation sets; the test set
remains untouched until the selected model is evaluated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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


TARGET_COLUMN = "residual"
SOURCE_FILE_COLUMN = "__source_file_id"
EXCLUDED_COLUMNS = {
    "datetime", TARGET_COLUMN, "gim_vtec", "TEC_smooth", "TEC_raw", SOURCE_FILE_COLUMN,
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，包括输入/输出目录、Optuna试验数量、XGBoost参数范围等。"""
    parser = argparse.ArgumentParser(
        description="Optuna-tuned XGBoost regression for Jason residuals (7:1:2)."
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
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if args.optuna_timeout < 0:
        raise ValueError("--optuna-timeout cannot be negative")
    if args.min_estimators < 1 or args.max_estimators < args.min_estimators:
        raise ValueError("Estimator range is invalid")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds cannot be negative")
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


def clean_and_validate(
    data: pd.DataFrame, requested_altitude: float | None
) -> tuple[pd.DataFrame, list[str], float, dict[str, float]]:
    """清理数据（去除缺失值和异常值），验证数据完整性，固定高度值。"""
    required = {TARGET_COLUMN, "datetime", "lat", "lon", "alt"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions")
    non_numeric = [
        column for column in [*feature_columns, TARGET_COLUMN]
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise TypeError(f"Model columns must be numeric: {non_numeric}")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    rows_before = len(data)
    data = data.dropna(axis=0, how="any").reset_index(drop=True)
    log(f"Rows before cleaning: {rows_before:,}")
    log(f"Rows removed: {rows_before - len(data):,}; retained: {len(data):,}")
    if len(data) < 10:
        raise ValueError(f"Too few valid rows for a 7:1:2 split: {len(data)}")

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
    log(
        "Original altitude range: "
        f"{original_altitude['minimum']:.6f} to {original_altitude['maximum']:.6f}; "
        f"fixed at {fixed_altitude:.6f}"
    )
    return data, feature_columns, fixed_altitude, original_altitude


def split_by_time_period(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """按时间顺序划分数据集：前70%训练、中间10%验证、后20%测试（最后一个时间段）。"""
    # 确保数据按时间排序
    sorted_data = data.sort_values("datetime").reset_index(drop=True)
    total = len(sorted_data)
    
    # 按时间比例划分
    train_idx = int(0.7 * total)
    val_idx = int(0.8 * total)
    
    splits = {
        "train": np.arange(0, train_idx),
        "validation": np.arange(train_idx, val_idx),
        "test": np.arange(val_idx, total)
    }
    
    log(
        "Dataset split (time-based): "
        + ", ".join(
            f"{name}={len(index):,} ({len(index) / total:.2%})"
            for name, index in splits.items()
        )
    )
    
    # 打印时间范围
    date_train_start = sorted_data.iloc[splits["train"][0]]["datetime"]
    date_train_end = sorted_data.iloc[splits["train"][-1]]["datetime"]
    date_val_start = sorted_data.iloc[splits["validation"][0]]["datetime"]
    date_val_end = sorted_data.iloc[splits["validation"][-1]]["datetime"]
    date_test_start = sorted_data.iloc[splits["test"][0]]["datetime"]
    date_test_end = sorted_data.iloc[splits["test"][-1]]["datetime"]
    
    log(f"Train period: {date_train_start} to {date_train_end}")
    log(f"Validation period: {date_val_start} to {date_val_end}")
    log(f"Test period: {date_test_start} to {date_test_end}")
    
    return splits


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
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "error_std": float(np.std(y_pred - y_true)),
    }


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
        columns=[SOURCE_FILE_COLUMN], errors="ignore"
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

    files = find_csv_files(args.input_dir, args.pattern, args.recursive)
    log(f"Found {len(files):,} CSV files")
    data = load_csv_files(files, args.csv_engine, args.read_batch_size)
    data, feature_columns, fixed_altitude, original_altitude = clean_and_validate(
        data, args.fixed_altitude
    )
    log(f"Model features ({len(feature_columns)}): {feature_columns}")
    if args.save_eda:
        save_eda_figures(data, args.output_dir)

    indices = split_by_time_period(data)
    # 按时间排序数据以匹配索引，保存原始行号映射
    data_sorted = data.sort_values("datetime").reset_index(drop=False)
    original_indices = data_sorted["index"].values
    data_sorted = data_sorted.reset_index(drop=True)
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
    for name, split_index in indices.items():
        prediction = model.predict(x_values[split_index])
        predictions[name] = prediction
        evaluations[name] = regression_metrics(y_values[split_index], prediction)
        log(
            f"{name}: RMSE={evaluations[name]['rmse']:.6f}, "
            f"MAE={evaluations[name]['mae']:.6f}, R2={evaluations[name]['r2']:.6f}"
        )

    test_index = indices["test"]
    pd.DataFrame({
        "original_row_index": original_indices[test_index],
        "true_residual": y_values[test_index],
        "predicted_residual": predictions["test"],
        "prediction_error": predictions["test"] - y_values[test_index],
    }).to_csv(args.output_dir / "test_predictions.csv", index=False)
    save_test_figure(
        y_values[test_index], predictions["test"],
        args.output_dir / "test_performance.png",
        args.max_scatter_points, args.model_seed,
    )

    metrics = {
        "input_directory": str(args.input_dir.resolve()),
        "input_file_count": len(files),
        "valid_row_count": len(data),
        "split_method": "time-based (chronological order)",
        "split_ratio": {"train": 0.7, "validation": 0.1, "test": 0.2},
        "split_rows": {name: len(value) for name, value in indices.items()},
        "time_periods": {
            "train": {
                "start": str(data_sorted.iloc[indices["train"][0]]["datetime"]),
                "end": str(data_sorted.iloc[indices["train"][-1]]["datetime"])
            },
            "validation": {
                "start": str(data_sorted.iloc[indices["validation"][0]]["datetime"]),
                "end": str(data_sorted.iloc[indices["validation"][-1]]["datetime"])
            },
            "test": {
                "start": str(data_sorted.iloc[indices["test"][0]]["datetime"]),
                "end": str(data_sorted.iloc[indices["test"][-1]]["datetime"])
            }
        },
        "model_seed": args.model_seed,
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
