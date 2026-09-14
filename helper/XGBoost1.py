#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMF3-FC ZTD精化模型 - XGBoost + Optuna超参数调优
替代Li et al. (2025)论文中的FNN模型

数据集划分: 训练集:验证集:测试集 = 7:1:2
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==================== 配置参数 ====================
CONFIG = {
    # 数据路径
    'training_data': r"/share/home/u23114/tj23114/data/zhongdian_data/VMF3_evaluation_results/fnn_training_data.csv",
    'output_dir': r"/share/home/u23114/tj23114/data/zhongdian_data/xgboost_model",
    
    # 数据集划分比例
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
    
    # Optuna超参数调优
    'n_trials': 100,  # 调优试验次数
    'timeout': 3600,  # 超时时间（秒）
    
    # 随机种子
    'random_seed': 42,
    
    # 是否绘图
    'plot_results': True,
}


# ==================== 数据加载与预处理 ====================

def load_and_preprocess_data(filepath):
    """
    加载并预处理训练数据
    
    输入特征: latitude, longitude, height, doy, utc, product_ztd
    输出特征: gnss_ztd
    """
    print("加载数据...")
    df = pd.read_csv(filepath)
    
    print(f"  总样本数: {len(df):,}")
    print(f"  特征列: {list(df.columns)}")
    
    # 检查缺失值
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  发现 {missing} 个缺失值，已删除")
        df = df.dropna()
    
    # 定义特征和标签
    feature_cols = ['latitude', 'longitude', 'height', 'doy', 'utc', 'product_ztd']
    target_col = 'gnss_ztd'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"  特征维度: {X.shape}")
    print(f"  标签维度: {y.shape}")
    
    return X, y, feature_cols


def split_dataset(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42):
    """
    划分数据集: 训练集:验证集:测试集 = 7:1:2
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 首先划分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed
    )
    
    # 从剩余数据中划分训练集和验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_seed
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  验证集: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  测试集: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==================== Optuna超参数调优 ====================

def create_objective(X_train, y_train, X_val, y_val):
    """
    创建Optuna目标函数
    """
    def objective(trial):
        # 定义超参数搜索空间
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'tree_method': 'hist',  # 使用histogram方法加速
            'random_state': CONFIG['random_seed'],
            'verbosity': 0,
            
            # 调优参数
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        # 训练模型
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 在验证集上评估
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred)) * 1000  # 转为mm
        
        return rmse
    
    return objective


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, timeout=3600):
    """
    使用Optuna进行超参数调优
    """
    print("\n" + "="*60)
    print("Optuna超参数调优")
    print("="*60)
    
    # 创建study
    sampler = TPESampler(seed=CONFIG['random_seed'])
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='xgboost_ztd_refinement'
    )
    
    # 创建目标函数
    objective = create_objective(X_train, y_train, X_val, y_val)
    
    # 进度回调
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"  Trial {trial.number}: RMSE = {trial.value:.4f} mm")
    
    print(f"开始调优 (最多 {n_trials} 次试验, 超时 {timeout}s)...")
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n调优完成!")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  完成试验数: {len(study.trials)}")
    print(f"  最佳RMSE: {study.best_value:.4f} mm")
    print(f"\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


# ==================== 模型训练与评估 ====================

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    使用最佳超参数训练最终模型
    """
    print("\n" + "="*60)
    print("训练最终模型")
    print("="*60)
    
    # 构建完整参数
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'random_state': CONFIG['random_seed'],
        'verbosity': 1,
        **best_params
    }
    
    # 合并训练集和验证集进行最终训练
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    print(f"训练样本数: {len(X_train_full):,}")
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_full, y_train_full, verbose=True)
    
    print("模型训练完成!")
    
    return model


def evaluate_model(model, X_test, y_test, feature_cols):
    """
    评估模型性能并计算精化效果
    """
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 获取原始Product_ZTD（第6个特征）
    product_ztd = X_test[:, 5]  # product_ztd是第6个特征
    
    # 计算精化前后的差异（转为mm）
    diff_before = (product_ztd - y_test) * 1000  # 精化前: Product_ZTD - GNSS_ZTD
    diff_after = (y_pred - y_test) * 1000  # 精化后: Refined_ZTD - GNSS_ZTD
    
    # 统计指标
    def calc_metrics(diff):
        return {
            'n': len(diff),
            'mean': np.mean(diff),
            'std': np.std(diff),
            'mae': np.mean(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff**2)),
            'max': np.max(np.abs(diff)),
            'min': np.min(np.abs(diff))
        }
    
    metrics_before = calc_metrics(diff_before)
    metrics_after = calc_metrics(diff_after)
    
    # 计算改进百分比
    rmse_improvement = (metrics_before['rmse'] - metrics_after['rmse']) / metrics_before['rmse'] * 100
    mae_improvement = (metrics_before['mae'] - metrics_after['mae']) / metrics_before['mae'] * 100
    std_improvement = (metrics_before['std'] - metrics_after['std']) / metrics_before['std'] * 100
    
    # 打印结果
    print(f"\n测试集样本数: {len(y_test):,}")
    print(f"\n{'指标':<12} {'精化前(mm)':>12} {'精化后(mm)':>12} {'改进(%)':>12}")
    print("-"*52)
    print(f"{'MEAN':<12} {metrics_before['mean']:>12.4f} {metrics_after['mean']:>12.4f} {'-':>12}")
    print(f"{'STD':<12} {metrics_before['std']:>12.4f} {metrics_after['std']:>12.4f} {std_improvement:>12.2f}")
    print(f"{'MAE':<12} {metrics_before['mae']:>12.4f} {metrics_after['mae']:>12.4f} {mae_improvement:>12.2f}")
    print(f"{'RMSE':<12} {metrics_before['rmse']:>12.4f} {metrics_after['rmse']:>12.4f} {rmse_improvement:>12.2f}")
    print("-"*52)
    
    # R²分数
    r2 = r2_score(y_test, y_pred)
    print(f"\nR² Score: {r2:.6f}")
    
    # 特征重要性
    print(f"\n特征重要性:")
    importance = model.feature_importances_
    for i, (col, imp) in enumerate(sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)):
        print(f"  {col}: {imp:.4f}")
    
    results = {
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'rmse_improvement': rmse_improvement,
        'mae_improvement': mae_improvement,
        'std_improvement': std_improvement,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'product_ztd': product_ztd,
        'diff_before': diff_before,
        'diff_after': diff_after
    }
    
    return results


# ==================== 可视化 ====================

def plot_results(results, output_dir):
    """
    绘制评估结果图
    """
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    y_test = results['y_test']
    y_pred = results['y_pred']
    product_ztd = results['product_ztd']
    diff_before = results['diff_before']
    diff_after = results['diff_after']
    
    # 1. 精化前散点图
    ax1 = axes[0, 0]
    ax1.scatter(y_test, product_ztd, alpha=0.1, s=1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('GNSS ZTD (m)')
    ax1.set_ylabel('Product ZTD (m)')
    ax1.set_title(f'Before Refinement\nRMSE={results["metrics_before"]["rmse"]:.2f}mm')
    ax1.set_aspect('equal')
    
    # 2. 精化后散点图
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred, alpha=0.1, s=1)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('GNSS ZTD (m)')
    ax2.set_ylabel('Refined ZTD (m)')
    ax2.set_title(f'After Refinement\nRMSE={results["metrics_after"]["rmse"]:.2f}mm')
    ax2.set_aspect('equal')
    
    # 3. RMSE对比柱状图
    ax3 = axes[0, 2]
    metrics = ['RMSE', 'MAE', 'STD']
    before_vals = [results['metrics_before']['rmse'], results['metrics_before']['mae'], results['metrics_before']['std']]
    after_vals = [results['metrics_after']['rmse'], results['metrics_after']['mae'], results['metrics_after']['std']]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax3.bar(x - width/2, before_vals, width, label='Before', color='coral')
    bars2 = ax3.bar(x + width/2, after_vals, width, label='After', color='steelblue')
    ax3.set_ylabel('Value (mm)')
    ax3.set_title('Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.bar_label(bars1, fmt='%.2f', fontsize=8)
    ax3.bar_label(bars2, fmt='%.2f', fontsize=8)
    
    # 4. 精化前残差直方图
    ax4 = axes[1, 0]
    ax4.hist(diff_before, bins=100, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Residual (mm)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Before: μ={np.mean(diff_before):.2f}, σ={np.std(diff_before):.2f}')
    ax4.set_xlim(-100, 100)
    
    # 5. 精化后残差直方图
    ax5 = axes[1, 1]
    ax5.hist(diff_after, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', lw=2)
    ax5.set_xlabel('Residual (mm)')
    ax5.set_ylabel('Density')
    ax5.set_title(f'After: μ={np.mean(diff_after):.2f}, σ={np.std(diff_after):.2f}')
    ax5.set_xlim(-100, 100)
    
    # 6. 残差对比（叠加直方图）
    ax6 = axes[1, 2]
    ax6.hist(diff_before, bins=100, density=True, alpha=0.5, label='Before', color='coral')
    ax6.hist(diff_after, bins=100, density=True, alpha=0.5, label='After', color='steelblue')
    ax6.axvline(x=0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Residual (mm)')
    ax6.set_ylabel('Density')
    ax6.set_title('Residual Distribution Comparison')
    ax6.legend()
    ax6.set_xlim(-100, 100)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(output_dir, 'xgboost_evaluation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {plot_path}")
    
    return plot_path


def plot_optuna_results(study, output_dir):
    """
    绘制Optuna调优结果
    """
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import plotly.io as pio
        
        # 优化历史
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(output_dir, 'optuna_history.png'))
        
        # 参数重要性
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(output_dir, 'optuna_importance.png'))
        
        print(f"  Optuna图表已保存")
    except Exception as e:
        print(f"  Optuna图表生成失败: {e}")


# ==================== 保存结果 ====================

def save_results(model, study, results, feature_cols, output_dir):
    """
    保存模型和结果
    """
    print("\n保存结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存模型
    model_path = os.path.join(output_dir, 'xgboost_ztd_model.json')
    model.save_model(model_path)
    print(f"  模型已保存: {model_path}")
    
    # 2. 保存最佳超参数
    params_path = os.path.join(output_dir, 'best_params.json')
    import json
    with open(params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"  超参数已保存: {params_path}")
    
    # 3. 保存评估结果
    eval_results = {
        'before_refinement': {
            'mean_mm': results['metrics_before']['mean'],
            'std_mm': results['metrics_before']['std'],
            'mae_mm': results['metrics_before']['mae'],
            'rmse_mm': results['metrics_before']['rmse']
        },
        'after_refinement': {
            'mean_mm': results['metrics_after']['mean'],
            'std_mm': results['metrics_after']['std'],
            'mae_mm': results['metrics_after']['mae'],
            'rmse_mm': results['metrics_after']['rmse']
        },
        'improvement': {
            'rmse_percent': results['rmse_improvement'],
            'mae_percent': results['mae_improvement'],
            'std_percent': results['std_improvement']
        },
        'r2_score': results['r2'],
        'n_test_samples': len(results['y_test']),
        'feature_importance': dict(zip(feature_cols, model.feature_importances_.tolist()))
    }
    
    eval_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"  评估结果已保存: {eval_path}")
    
    # 4. 保存Optuna study
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    joblib.dump(study, study_path)
    print(f"  Optuna study已保存: {study_path}")
    
    return model_path, params_path, eval_path


# ==================== 预测函数（供后续使用） ====================

def predict_refined_ztd(model, latitude, longitude, height, doy, utc, product_ztd):
    """
    使用训练好的模型预测精化后的ZTD
    
    Parameters:
    -----------
    model : XGBRegressor
        训练好的模型
    latitude : float or array
        纬度
    longitude : float or array
        经度
    height : float or array
        高程（m）
    doy : int or array
        年积日
    utc : int or array
        UTC小时
    product_ztd : float or array
        VMF3产品ZTD（m）
    
    Returns:
    --------
    refined_ztd : float or array
        精化后的ZTD（m）
    """
    # 构建特征矩阵
    if np.isscalar(latitude):
        X = np.array([[latitude, longitude, height, doy, utc, product_ztd]])
    else:
        X = np.column_stack([latitude, longitude, height, doy, utc, product_ztd])
    
    # 预测
    refined_ztd = model.predict(X)
    
    return refined_ztd[0] if np.isscalar(latitude) else refined_ztd


# ==================== 主程序 ====================

def main():
    start_time = time.time()
    
    print("="*60)
    print("VMF3-FC ZTD精化模型 - XGBoost + Optuna")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. 加载数据
    X, y, feature_cols = load_and_preprocess_data(CONFIG['training_data'])
    
    # 2. 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y,
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        test_ratio=CONFIG['test_ratio'],
        random_seed=CONFIG['random_seed']
    )
    
    # 3. 超参数调优
    study = optimize_hyperparameters(
        X_train, y_train, X_val, y_val,
        n_trials=CONFIG['n_trials'],
        timeout=CONFIG['timeout']
    )
    
    # 4. 训练最终模型
    model = train_final_model(X_train, y_train, X_val, y_val, study.best_params)
    
    # 5. 评估模型
    results = evaluate_model(model, X_test, y_test, feature_cols)
    
    # 6. 保存结果
    save_results(model, study, results, feature_cols, CONFIG['output_dir'])
    
    # 7. 可视化
    if CONFIG['plot_results']:
        plot_results(results, CONFIG['output_dir'])
        plot_optuna_results(study, CONFIG['output_dir'])
    
    # 打印总结
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"\n精化效果总结:")
    print(f"  RMSE: {results['metrics_before']['rmse']:.2f}mm → {results['metrics_after']['rmse']:.2f}mm")
    print(f"  改进: {results['rmse_improvement']:.2f}%")
    print(f"\n模型保存位置: {CONFIG['output_dir']}")
    
    return model, study, results


if __name__ == '__main__':
    model, study, results = main()