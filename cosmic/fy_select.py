import os
import sys
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from collections import defaultdict
import multiprocessing as mp
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==========================================================
# 1. 参数配置
# ==========================================================
NM_F2_MIN, NM_F2_MAX = 2e10, 2e12   # 单位: el/m^3
HM_F2_MIN, HM_F2_MAX = 180.0, 450.0 # 单位: km

# 论文形态学规则参数 (辽宁工程技术大学标准)
MD_MAX = 1.0             
DELTA_MAX = 0.1          
GRADIENT_MAX = 0.0       

# --- 学长建议新增：切点轨迹跨度阈值 (弧度筛选) ---
# 逻辑：切点在经纬度上的漂移范围若超过5度，则球对称假设失效
DRIFT_MAX = 50

# Hampel 滤波参数 (拔刺)
HAMPEL_WINDOW_KM = 50.0  
HAMPEL_SIGMA = 2.5       

# ==========================================================
# 2. 核心功能函数
# ==========================================================

def apply_hampel_filter(alt, ne):
    """
    Hampel 滤波器：识别并剔除剖面中的突发噪声点点
    """
    n = len(ne)
    if n < 10: return ne, 0
    new_ne = np.array(ne, dtype=float).copy()
    avg_spacing = np.mean(np.diff(alt))
    half_window = int((HAMPEL_WINDOW_KM / 2) / avg_spacing)
    if half_window < 3: half_window = 3
    
    removed_count = 0
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window)
        window_data = ne[start:end]
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))
        sigma = 1.4826 * max(mad, median * 0.05, 1e7)
        if np.abs(ne[i] - median) > HAMPEL_SIGMA * sigma:
            new_ne[i] = np.nan
            removed_count += 1
    return new_ne, removed_count

def qc_single_profile(fpath):
    """
    单剖面综合质控逻辑
    """
    try:
        with xr.open_dataset(fpath) as ds:
            # --- 步骤 1: 学长建议的“弧度”筛选 (切点轨迹跨度) ---
            if 'lat_tp' in ds and 'lon_tp' in ds:
                lats_tp = ds['lat_tp'].values.flatten()
                lons_tp = ds['lon_tp'].values.flatten()
                
                # 计算切点轨迹的最大地理跨度 (Drift)
                lat_span = np.nanmax(lats_tp) - np.nanmin(lats_tp)
                lon_span = np.nanmax(lons_tp) - np.nanmin(lons_tp)
                total_drift = np.sqrt(lat_span**2 + lon_span**2)
                
                if total_drift > DRIFT_MAX:
                    return fpath, "drift_too_large", 0
            else:
                return fpath, "missing_tp_data", 0

            # --- 步骤 2: 全局负值硬拦截 --- 
            raw_alt = ds['MSL_alt'].values.flatten()
            ne_raw = ds['elec_Dens'].values.flatten()
            if np.any(ne_raw < -100): 
                return fpath, "negative_value", 0

            # 初始清洗
            valid = np.isfinite(raw_alt) & np.isfinite(ne_raw)
            alt_v, ne_v = raw_alt[valid], ne_raw[valid]
            if len(alt_v) < 30: return fpath, "too_short", 0
            
            # 单位换算: cm^-3 -> m^-3
            if np.max(ne_v) < 1e8: ne_v *= 1e6 
            idx = np.argsort(alt_v)
            alt_v, ne_v = alt_v[idx], ne_v[idx]

            # --- 步骤 3: Hampel 拔刺 ---
            ne_filtered, n_rem = apply_hampel_filter(alt_v, ne_v)
            if n_rem / len(ne_v) > 0.3: return fpath, "too_noisy", n_rem
            
            # --- 步骤 4: 基础物理检查 (NmF2/hmF2) --- 
            clean_mask = ~np.isnan(ne_filtered)
            alt_c, ne_c = alt_v[clean_mask], ne_filtered[clean_mask]
            nm_f2 = np.max(ne_c)
            hm_f2 = alt_c[np.argmax(ne_c)]
            
            if not (NM_F2_MIN <= nm_f2 <= NM_F2_MAX): return fpath, "NmF2_limit", n_rem
            if not (HM_F2_MIN <= hm_f2 <= HM_F2_MAX): return fpath, "hmF2_limit", n_rem
            
            # --- 步骤 5: 论文形态学规则 ---
            ne_interp = np.interp(alt_v, alt_c, ne_c)
            try:
                ne_smooth = savgol_filter(ne_interp, 35, 2)
            except: return fpath, "smooth_fail", n_rem
            
            # 1. 顶部梯度 (Gradient)
            idx_490 = np.argmin(np.abs(alt_v - 490.0))
            idx_420 = np.argmin(np.abs(alt_v - 420.0))
            if (ne_smooth[idx_490] - ne_smooth[idx_420]) / 70.0 > GRADIENT_MAX:
                return fpath, "gradient_error", n_rem
            
            # 2. 平均偏差 MD
            n_points = np.sum(clean_mask)
            md = np.sum(np.abs(ne_v[clean_mask] - ne_smooth[clean_mask]) / (n_points * ne_smooth[clean_mask]))
            if md > MD_MAX: return fpath, "MD_error", n_rem
            
            # 3. 噪声水平 Delta
            mask_300 = (alt_v >= 300.0) & clean_mask
            k = np.sum(mask_300)
            if k > 5:
                diff_sq = np.sum((ne_v[mask_300] - ne_smooth[mask_300])**2)
                delta = np.sqrt(diff_sq / (k * (nm_f2**2)))
                if delta > DELTA_MAX: return fpath, "Delta_error", n_rem

            return fpath, "pass", n_rem

    except Exception:
        return fpath, "file_error", 0

# ==========================================================
# 3. 主进程逻辑
# ==========================================================

def process_fy_data(root_dir):
    print(f"🚀 启动增强质控 (集成学长建议：弧度筛选 + 全局负值拦截)...")
    
    # 扫描所有子目录下的 NC 文件
    all_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".nc"):
                all_files.append(os.path.join(root, f))

    total_count = len(all_files)
    stat = defaultdict(int)
    passed_files = []
    total_spikes = 0

    # 使用并行计算加速处理
    cpu_to_use = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=cpu_to_use) as pool:
        # 结果汇总
        from tqdm import tqdm
        for fpath, result, n_rem in tqdm(pool.imap_unordered(qc_single_profile, all_files), total=total_count):
            stat[result] += 1
            total_spikes += n_rem
            if result == "pass": passed_files.append(fpath)

    # 保存通过清单，作为下一步“比例法”外推的输入
    output_file = r"F:\vs_studio_code\bs\results\fy3e_qc_A1_ A4_passed.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as out:
        for p in passed_files: out.write(p + "\n")

    # 输出增强版拦截报告
    print("\n" + "="*50)
    print(f"{'质量控制拦截报告 (FY-3E v5)':^46}")
    print("="*50)
    print(f" 1. 切点轨迹跨度过大 (Drift>5°)  : {stat['drift_too_large']}")
    print(f" 2. 包含负值数据 (Negative Val)  : {stat['negative_value']}")
    print(f" 3. 顶部梯度异常 (Gradient>0)    : {stat['gradient_error']}")
    print(f" 4. 平均偏差异常 (MD>1.0)       : {stat['MD_error']}")
    print(f" 5. 噪声水平超标 (Delta>0.1)     : {stat['Delta_error']}")
    print(f" 6. NmF2/hmF2 物理超限          : {stat['NmF2_limit'] + stat['hmF2_limit']}")
    print("-" * 50)
    print(f" 最终通过数: {len(passed_files)} / 总数: {total_count}")
    print(f" 最终通过率: {(len(passed_files)/total_count*100):.2f}%")
    print("="*50)

if __name__ == "__main__":
    # 配置风云数据存放的根目录
    TARGET_DIR = r"F:\bs\FY2024\FY3" 
    mp.freeze_support()
    process_fy_data(TARGET_DIR)