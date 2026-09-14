# 该文件主要是来做特征工程
import os
import glob
import numpy as np
import pandas as pd
from apexpy import Apex


# 各指数允许插值跨越的最大时间缺口。
# F10.7 通常为日值；Dst 通常为小时值；Kp 通常为 3 小时值。
# Kp 取 30 小时是为了能处理“整日缺失、前后仍有数据”的情况，同时避免
# 将相隔数天甚至更久的数据直接连接起来。
MAX_INTERPOLATION_GAP = {
    'f10.7_Index': pd.Timedelta(hours=72),
    'Dst_Index': pd.Timedelta(hours=6),
    'Kp_Index': pd.Timedelta(hours=30),
}

# ==========================================
# 1. 地磁经纬度转换函数 (准偶极坐标 QD)
# ==========================================
def geo_to_qd(df):
    """
    使用 apexpy 根据地理经纬度、高度(alt)和时间(datetime)计算准偶极坐标 (QD Latitude & Longitude)
    """
    lats = df['lat'].values
    lons = df['lon'].values
    alts = df['alt'].values
    
    ref_time = df['datetime'].iloc[0]
    apex_builder = Apex(date=ref_time)
    
    # 如果 alt 单位是米 (m)，请使用 alts / 1000.0；若已经是 km 则保持 alts
    qd_lat, qd_lon = apex_builder.geo2qd(lats, lons, height=alts)
    return qd_lat, qd_lon


# ==========================================
# 2. 读取并预处理指数参考 CSV 文件 (带 NaN 缺失值处理)
# ==========================================
def load_index_data(index_csv_path):
    df_idx = pd.read_csv(index_csv_path)
    
    # 自动识别与重命名表头
    col_map = {}
    for col in df_idx.columns:
        c_lower = col.lower()
        if 'year' in c_lower: col_map[col] = 'Year'
        elif 'decimal' in c_lower: col_map[col] = 'Decimal_Day'
        elif 'hour' in c_lower: col_map[col] = 'Hour'
        elif 'f10' in c_lower: col_map[col] = 'f10.7_Index'
        elif 'dst' in c_lower: col_map[col] = 'Dst_Index'
        elif 'kp' in c_lower: col_map[col] = 'Kp_Index'
        
    df_idx = df_idx.rename(columns=col_map)
    
    required_cols = {
        'Year', 'Decimal_Day', 'Hour',
        'f10.7_Index', 'Dst_Index', 'Kp_Index'
    }
    missing_cols = required_cols.difference(df_idx.columns)
    if missing_cols:
        raise ValueError(f"指数文件缺少必要列: {sorted(missing_cols)}")

    # 先转为数值，文本、空字符串等异常内容统一视为缺失值。
    numeric_cols = list(required_cols)
    df_idx[numeric_cols] = df_idx[numeric_cols].apply(
        pd.to_numeric, errors='coerce'
    )

    invalid_time = df_idx[['Year', 'Decimal_Day', 'Hour']].isna().any(axis=1)
    if invalid_time.any():
        print(f"警告: 指数文件中有 {invalid_time.sum()} 行时间字段无效，已跳过。")
        df_idx = df_idx.loc[~invalid_time].copy()

    # 构建标准 datetime 时间轴。Decimal_Day 从 1 开始，Hour 从 0 开始。
    base_dates = pd.to_datetime(
        df_idx['Year'].astype(int).astype(str) + '-01-01', errors='coerce'
    )
    df_idx['datetime'] = base_dates + \
                         pd.to_timedelta(df_idx['Decimal_Day'] - 1, unit='D') + \
                         pd.to_timedelta(df_idx['Hour'], unit='h')

    invalid_datetime = df_idx['datetime'].isna()
    if invalid_datetime.any():
        print(f"警告: 指数文件中有 {invalid_datetime.sum()} 行日期无法解析，已跳过。")
        df_idx = df_idx.loc[~invalid_datetime].copy()

    # OMNI 的填充值必须逐列处理，不能混用。例如 99.9 是合法的 F10.7，
    # 但 99 是 Kp 的缺失码，99999 是 Dst 的缺失码。
    df_idx['f10.7_Index'] = df_idx['f10.7_Index'].replace(999.9, np.nan)
    df_idx['Dst_Index'] = df_idx['Dst_Index'].replace(99999, np.nan)
    df_idx['Kp_Index'] = df_idx['Kp_Index'].replace(99, np.nan)

    # 兼容直接读取的 OMNI 原始 Kp（单位为 Kp*10）和已预处理的 Kp（0~9）。
    kp_valid = df_idx['Kp_Index'].dropna()
    if not kp_valid.empty and kp_valid.quantile(0.95) > 9:
        df_idx['Kp_Index'] = df_idx['Kp_Index'] / 10.0

    # 将明显超出物理/格式范围的值视为异常，而不是拿来参与插值。
    valid_ranges = {
        'f10.7_Index': (0.0, 500.0),
        'Dst_Index': (-2500.0, 1000.0),
        'Kp_Index': (0.0, 9.0),
    }
    for col, (lower, upper) in valid_ranges.items():
        invalid_value = df_idx[col].notna() & ~df_idx[col].between(lower, upper)
        if invalid_value.any():
            print(f"警告: {col} 有 {invalid_value.sum()} 个异常值，已按缺失值处理。")
            df_idx.loc[invalid_value, col] = np.nan

    target_cols = ['f10.7_Index', 'Dst_Index', 'Kp_Index']

    # 同一时刻若出现重复记录，用中位数合并，避免重复 x 坐标影响插值。
    df_idx = (
        df_idx[['datetime', *target_cols]]
        .groupby('datetime', as_index=False, sort=True)
        .median(numeric_only=True)
    )

    return df_idx


# ==========================================
# 辅助函数：安全一维线性插值 (防 NaN 扩散)
# ==========================================
def safe_interp(target_x, x_arr, y_arr, max_gap=None):
    """
    按真实时间做一维线性插值。

    仅在目标时刻两侧都有有效观测，且两侧观测间隔不超过 max_gap 时插值；
    不在数据首尾之外外推。这样既能填补允许范围内的缺失，又不会跨越过长
    的空档或用首尾值无限填充。
    """
    target_x = np.asarray(target_x, dtype=np.float64)
    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    result = np.full(target_x.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(valid_mask):
        return result

    valid_x = x_arr[valid_mask]
    valid_y = y_arr[valid_mask]
    order = np.argsort(valid_x)
    valid_x = valid_x[order]
    valid_y = valid_y[order]

    positions = np.searchsorted(valid_x, target_x, side='left')

    # 目标时间正好有原始有效值时，始终保留该原值。
    clipped_positions = np.minimum(positions, len(valid_x) - 1)
    exact = (positions < len(valid_x)) & (
        valid_x[clipped_positions] == target_x
    )
    result[exact] = valid_y[clipped_positions[exact]]

    between = (~exact) & (positions > 0) & (positions < len(valid_x))
    if not np.any(between):
        return result

    between_idx = np.flatnonzero(between)
    right_pos = positions[between]
    left_pos = right_pos - 1
    gaps = valid_x[right_pos] - valid_x[left_pos]

    if max_gap is None:
        allowed = np.ones(gaps.shape, dtype=bool)
    else:
        max_gap_seconds = pd.Timedelta(max_gap).total_seconds()
        allowed = gaps <= max_gap_seconds

    if np.any(allowed):
        selected = between_idx[allowed]
        left = left_pos[allowed]
        right = right_pos[allowed]
        weights = (
            (target_x[selected] - valid_x[left])
            / (valid_x[right] - valid_x[left])
        )
        result[selected] = valid_y[left] + weights * (
            valid_y[right] - valid_y[left]
        )

    return result


# ==========================================
# 3. 批量处理函数
# ==========================================
def process_all_files(input_folder, index_csv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    print("正在加载与解析指数参考数据...")
    df_idx = load_index_data(index_csv_path)
    
    # 转换为 Unix 时间戳 (秒)
    idx_timestamps = df_idx['datetime'].astype('int64').to_numpy() / 10**9
    f107_vals = df_idx['f10.7_Index'].values
    dst_vals  = df_idx['Dst_Index'].values
    kp_vals   = df_idx['Kp_Index'].values
    
    file_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(file_pattern)
    print(f"共找到 {len(csv_files)} 个待处理 CSV 文件。")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"正在处理: {filename} ...")
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        invalid_target_time = df['datetime'].isna()
        if invalid_target_time.any():
            print(
                f"警告: {filename} 中有 {invalid_target_time.sum()} 行 datetime 无效，"
                "这些行的指数结果将保留为 NaN。"
            )
        
        # ----------------------------------------------------
        # 特征 1：根据【图一】公式计算 4 个时间周期正余弦特征
        # ----------------------------------------------------
        hod = df['datetime'].dt.hour + \
              df['datetime'].dt.minute / 60.0 + \
              df['datetime'].dt.second / 3600.0
        
        # DOY 加上日内小数部分，保证时间连续平滑
        doy = df['datetime'].dt.dayofyear 
        
        df['HOD_s'] = np.sin(2 * np.pi * hod / 24.0)
        df['HOD_c'] = np.cos(2 * np.pi * hod / 24.0)
        df['DOY_s'] = np.sin(2 * np.pi * doy / 365.25)
        df['DOY_c'] = np.cos(2 * np.pi * doy / 365.25)
        
        # ----------------------------------------------------
        # 特征 2：安全插值 F10.7, Dst, Kp 指数
        # ----------------------------------------------------
        target_timestamps = df['datetime'].astype('int64').to_numpy() / 10**9
        target_timestamps[invalid_target_time.to_numpy()] = np.nan

        df['f10.7_Index'] = safe_interp(
            target_timestamps, idx_timestamps, f107_vals,
            MAX_INTERPOLATION_GAP['f10.7_Index']
        )
        df['Dst_Index'] = safe_interp(
            target_timestamps, idx_timestamps, dst_vals,
            MAX_INTERPOLATION_GAP['Dst_Index']
        )
        df['Kp_Index'] = safe_interp(
            target_timestamps, idx_timestamps, kp_vals,
            MAX_INTERPOLATION_GAP['Kp_Index']
        )

        for col in ('f10.7_Index', 'Dst_Index', 'Kp_Index'):
            missing_count = df[col].isna().sum()
            if missing_count:
                print(
                    f"警告: {filename} 的 {col} 有 {missing_count} 行无法可靠插值，"
                    "已保留为 NaN。"
                )
        
        # ----------------------------------------------------
        # 特征 3：计算地磁纬度 (mag_lat) 与 地磁经度 (mag_lon)
        # ----------------------------------------------------
        mag_lat, mag_lon = geo_to_qd(df)
        df['mag_lat'] = mag_lat
        df['mag_lon'] = mag_lon
        
        # ----------------------------------------------------
        # 输出保存
        # ----------------------------------------------------
        out_path = os.path.join(output_folder, filename)
        df.to_csv(out_path, index=False)
        print(f"成功导出到: {out_path}")

    print("所有文件处理完成！")


# ==========================================
# 4. 脚本入口
# ==========================================
if __name__ == '__main__':
    INPUT_FOLDER   = r"F:\FusedTec\Data\Jason2021\TecCSV2_redisual" 
    INDEX_CSV_PATH = r"F:\FusedTec\Data\Other\omni2_2021.dat.csv" 
    OUTPUT_FOLDER  = r"F:\FusedTec\Data\Jason2021\TecCSV2_redisual_CH" #表示特征
    
    process_all_files(INPUT_FOLDER, INDEX_CSV_PATH, OUTPUT_FOLDER)
