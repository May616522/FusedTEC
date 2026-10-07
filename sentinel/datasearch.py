import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. 参数设置
# ============================================================

file_path = (
    r"F:\FusedTec\Data\Sentinel2021"
    r"\S6A_P4_2__LR_RED__NT_041_188_20211226T142429_20211226T152042_G01.nc"
)

output_dir = Path(
    r"F:\FusedTec\Data\Sentinel2021\diagnostic_output"
)
output_dir.mkdir(parents=True, exist_ok=True)

SAVE_CSV = True


# ============================================================
# 2. 读取不同 group
# ============================================================

print("=" * 100)
print("读取 Sentinel-6 文件")
print("=" * 100)
print(file_path)

# data_01：公共1 Hz变量
ds = xr.open_dataset(
    file_path,
    group="data_01"
)

# data_01/ku：Ku-band相关变量
ds_ku = xr.open_dataset(
    file_path,
    group="data_01/ku"
)

print("\n" + "=" * 100)
print("data_01 基本信息")
print("=" * 100)
print(ds)

print("\n" + "=" * 100)
print("data_01/ku 基本信息")
print("=" * 100)
print(ds_ku)


# ============================================================
# 3. 查看变量列表
# ============================================================

print("\n" + "=" * 100)
print("data_01 变量")
print("=" * 100)

for var in ds.data_vars:
    print(var)

print("\n" + "=" * 100)
print("data_01/ku 变量")
print("=" * 100)

for var in ds_ku.data_vars:
    print(var)


# ============================================================
# 4. 提取公共变量
# ============================================================

time = ds["time"].values
lat = ds["latitude"].values
lon = ds["longitude"].values
altitude = ds["altitude"].values

# 经度 0~360 -> -180~180
lon_180 = (lon + 180) % 360 - 180


# ============================================================
# 5. 提取 TEC
# ============================================================

# xarray 默认已自动应用 NetCDF scale_factor
# 当前单位已经是 electrons/m^2
tec_m2 = ds["total_electron_content"].values

# 转成 TECU
tec_tecu = tec_m2 / 1e16


# ============================================================
# 6. 提取 Sentinel-6 电离层改正
# ============================================================

iono_cor_alt = ds["iono_cor_alt"].values

iono_cor_alt_filtered = (
    ds["iono_cor_alt_filtered"].values
)

iono_cor_alt_nr = (
    ds["iono_cor_alt_nr"].values
    if "iono_cor_alt_nr" in ds.variables
    else np.full(len(time), np.nan)
)

iono_cor_alt_filtered_nr = (
    ds["iono_cor_alt_filtered_nr"].values
    if "iono_cor_alt_filtered_nr" in ds.variables
    else np.full(len(time), np.nan)
)


# ============================================================
# 7. 提取 GIM-derived ionospheric correction
#    注意：位于 data_01/ku group
# ============================================================

if "iono_cor_gim" not in ds_ku.variables:
    raise KeyError(
        "在 data_01/ku 中没有找到 iono_cor_gim，请检查文件结构。"
    )

iono_cor_gim = ds_ku["iono_cor_gim"].values


# ============================================================
# 8. 检查维度是否一致
# ============================================================

print("\n" + "=" * 100)
print("维度检查")
print("=" * 100)

print(f"time长度                 : {len(time)}")
print(f"TEC长度                  : {len(tec_tecu)}")
print(f"iono_cor_alt长度         : {len(iono_cor_alt)}")
print(f"iono_cor_alt_filtered长度: {len(iono_cor_alt_filtered)}")
print(f"iono_cor_gim长度         : {len(iono_cor_gim)}")

if len(iono_cor_gim) != len(time):
    raise ValueError(
        "iono_cor_gim 与 data_01 的 time 长度不一致，"
        "不能直接按位置进行1 Hz匹配。"
    )


# ============================================================
# 9. 提取 QC 字段
# ============================================================

surface_type = ds["surface_classification_flag"].values
rain_flag = ds["rain_flag"].values
pass_direction = ds["pass_direction_flag"].values

manoeuvre_flag = (
    ds["manoeuvre_flag"].values
    if "manoeuvre_flag" in ds.variables
    else np.full(len(time), np.nan)
)

rad_sea_ice_flag = (
    ds["rad_sea_ice_flag"].values
    if "rad_sea_ice_flag" in ds.variables
    else np.full(len(time), np.nan)
)

amr_rain_flag = (
    ds["amr_rain_flag"].values
    if "amr_rain_flag" in ds.variables
    else np.full(len(time), np.nan)
)

rad_rain_flag = (
    ds["rad_rain_flag"].values
    if "rad_rain_flag" in ds.variables
    else np.full(len(time), np.nan)
)

distance_to_coast = (
    ds["distance_to_coast"].values
    if "distance_to_coast" in ds.variables
    else np.full(len(time), np.nan)
)


# ============================================================
# 10. 构建 DataFrame
# ============================================================

df = pd.DataFrame({
    "time": time,
    "lat": lat,
    "lon": lon_180,

    "altitude_m": altitude,
    "TEC_TECU": tec_tecu,

    "iono_cor_alt_m": iono_cor_alt,
    "iono_cor_alt_filtered_m": iono_cor_alt_filtered,
    "iono_cor_alt_nr_m": iono_cor_alt_nr,
    "iono_cor_alt_filtered_nr_m": iono_cor_alt_filtered_nr,

    "iono_cor_gim_m": iono_cor_gim,

    "surface_type": surface_type,
    "rain_flag": rain_flag,
    "amr_rain_flag": amr_rain_flag,
    "rad_rain_flag": rad_rain_flag,
    "rad_sea_ice_flag": rad_sea_ice_flag,

    "pass_direction": pass_direction,
    "manoeuvre_flag": manoeuvre_flag,

    "distance_to_coast_m": distance_to_coast
})


# ============================================================
# 11. Altimeter - GIM correction residual
# ============================================================

df["delta_iono_m"] = (
    df["iono_cor_alt_filtered_m"]
    - df["iono_cor_gim_m"]
)


# ============================================================
# 12. 基础输出
# ============================================================

print("\n" + "=" * 100)
print("前20条数据")
print("=" * 100)

print(df.head(20).to_string())


print("\n" + "=" * 100)
print("TEC统计")
print("=" * 100)

print(df["TEC_TECU"].describe())


print("\n" + "=" * 100)
print("iono_cor_gim统计")
print("=" * 100)

print(df["iono_cor_gim_m"].describe())

print(
    "\n有效 iono_cor_gim 数量：",
    df["iono_cor_gim_m"].notna().sum()
)


# ============================================================
# 13. 缺失值统计
# ============================================================

print("\n" + "=" * 100)
print("缺失值统计")
print("=" * 100)

missing_df = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100
})

print(missing_df.to_string())


# ============================================================
# 14. Surface type统计
# ============================================================

surface_name = {
    0: "open_ocean",
    1: "land",
    2: "continental_water",
    3: "aquatic_vegetation",
    4: "continental_ice_snow",
    5: "floating_ice",
    6: "salted_basin"
}

print("\n" + "=" * 100)
print("Surface Classification统计")
print("=" * 100)

surface_counts = (
    df["surface_type"]
    .value_counts()
    .sort_index()
)

for value, count in surface_counts.items():

    value_int = int(value)

    print(
        f"{value_int}: "
        f"{surface_name.get(value_int, 'unknown'):<25}"
        f"{count}"
    )


# ============================================================
# 15. Open Ocean + 有效TEC
# ============================================================

ocean = df[
    (df["surface_type"] == 0)
    &
    np.isfinite(df["TEC_TECU"])
    &
    (df["TEC_TECU"] > 0)
].copy()


print("\n" + "=" * 100)
print("Open Ocean TEC统计")
print("=" * 100)

print(f"全部记录数             : {len(df)}")
print(f"Open Ocean记录数       : {(df['surface_type'] == 0).sum()}")
print(f"Open Ocean有效TEC数量  : {len(ocean)}")

open_ocean_count = (
    df["surface_type"] == 0
).sum()

if open_ocean_count > 0:

    valid_rate = (
        len(ocean) /
        open_ocean_count *
        100
    )

    print(
        f"Open Ocean TEC有效率    : {valid_rate:.2f}%"
    )

print("\n海洋TEC统计：")
print(ocean["TEC_TECU"].describe())


# ============================================================
# 16. Altimeter filtered 与 GIM 共同有效
# ============================================================

compare = ocean[
    np.isfinite(
        ocean["iono_cor_alt_filtered_m"]
    )
    &
    np.isfinite(
        ocean["iono_cor_gim_m"]
    )
].copy()


print("\n" + "=" * 100)
print("Altimeter filtered correction vs GIM correction")
print("=" * 100)

print(
    f"共同有效点数量：{len(compare)}"
)


# ============================================================
# 17. 统计 R / Bias / STD / RMSE / MAE
# ============================================================

corr = np.nan
bias = np.nan
std = np.nan
rmse = np.nan
mae = np.nan

if len(compare) > 1:

    corr = np.corrcoef(
        compare["iono_cor_alt_filtered_m"],
        compare["iono_cor_gim_m"]
    )[0, 1]

    residual = (
        compare["delta_iono_m"].values
    )

    bias = np.mean(residual)

    std = np.std(
        residual,
        ddof=1
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    mae = np.mean(
        np.abs(
            residual
        )
    )

    print(
        f"相关系数 R : {corr:.6f}"
    )

    print(
        f"Bias       : {bias:.6f} m"
    )

    print(
        f"STD        : {std:.6f} m"
    )

    print(
        f"RMSE       : {rmse:.6f} m"
    )

    print(
        f"MAE        : {mae:.6f} m"
    )

    print("\nDelta iono统计：")
    print(
        compare["delta_iono_m"].describe()
    )

else:

    print(
        "共同有效点不足，无法计算统计量。"
    )


# ============================================================
# 18. TEC vs GIM correction
# ============================================================

tmp_tec_gim = ocean[
    np.isfinite(
        ocean["iono_cor_gim_m"]
    )
].copy()


corr_tec_gim = np.nan

if len(tmp_tec_gim) > 1:

    corr_tec_gim = np.corrcoef(
        tmp_tec_gim["TEC_TECU"],
        tmp_tec_gim["iono_cor_gim_m"]
    )[0, 1]

    print("\n" + "=" * 100)
    print("TEC vs GIM-derived correction")
    print("=" * 100)

    print(
        f"相关系数 R = {corr_tec_gim:.6f}"
    )


# ============================================================
# 19. Rain flag统计
# ============================================================

print("\n" + "=" * 100)
print("TEC按rain_flag统计")
print("=" * 100)

rain_stats = (
    ocean
    .groupby("rain_flag")["TEC_TECU"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        median="median",
        max="max"
    )
)

print(rain_stats)


# ============================================================
# 20. 连续1 Hz TEC差分
# ============================================================

diagnostic = ocean.copy()

diagnostic["dt_s"] = (
    diagnostic["time"]
    .diff()
    .dt.total_seconds()
)

diagnostic["dTEC"] = (
    diagnostic["TEC_TECU"]
    .diff()
    .abs()
)

# 只认为 <=2秒 是真正的连续轨道相邻点
diagnostic.loc[
    diagnostic["dt_s"] > 2,
    "dTEC"
] = np.nan


print("\n" + "=" * 100)
print("连续1 Hz TEC变化统计")
print("=" * 100)

print(
    diagnostic["dTEC"]
    .describe()
)


# ============================================================
# 21. 最大连续TEC变化20点
# ============================================================

jump_columns = [
    "time",
    "lat",
    "lon",
    "TEC_TECU",
    "dTEC",
    "dt_s",

    "rain_flag",
    "amr_rain_flag",
    "rad_rain_flag",
    "rad_sea_ice_flag",

    "iono_cor_alt_filtered_m",
    "iono_cor_gim_m",
    "delta_iono_m",

    "distance_to_coast_m"
]

top_jumps = (
    diagnostic
    .nlargest(
        20,
        "dTEC"
    )[jump_columns]
)


print("\n" + "=" * 100)
print("连续TEC变化最大的20个点")
print("=" * 100)

print(
    top_jumps.to_string(
        index=False
    )
)


# ============================================================
# 22. 第一版基础QC
# ============================================================

basic_mask = (
    (df["surface_type"] == 0)
    &
    np.isfinite(df["TEC_TECU"])
    &
    (df["TEC_TECU"] > 0)
)

df_basic_qc = df[
    basic_mask
].copy()


# ============================================================
# 23. 较严格QC候选
# ============================================================

strict_mask = (
    basic_mask
    &
    (
        df["manoeuvre_flag"].isna()
        |
        (df["manoeuvre_flag"] == 0)
    )
    &
    (
        df["rad_sea_ice_flag"].isna()
        |
        (df["rad_sea_ice_flag"] == 0)
    )
)

df_strict_candidate = df[
    strict_mask
].copy()


print("\n" + "=" * 100)
print("QC统计")
print("=" * 100)

print(
    f"原始记录数     : {len(df)}"
)

print(
    f"基础QC数量     : {len(df_basic_qc)}"
)

print(
    f"严格QC候选数量 : {len(df_strict_candidate)}"
)


# ============================================================
# 24. 图1：TEC时间序列
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.scatter(
    ocean["time"],
    ocean["TEC_TECU"],
    s=8
)

plt.xlabel("UTC Time")
plt.ylabel("TEC (TECU)")
plt.title("Sentinel-6 Altimeter TEC")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 25. 图2：Altimeter filtered vs GIM
# ============================================================

if len(compare) > 1:

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        compare["iono_cor_gim_m"],
        compare["iono_cor_alt_filtered_m"],
        s=10,
        alpha=0.5
    )

    vmin = min(
        compare["iono_cor_gim_m"].min(),
        compare["iono_cor_alt_filtered_m"].min()
    )

    vmax = max(
        compare["iono_cor_gim_m"].max(),
        compare["iono_cor_alt_filtered_m"].max()
    )

    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        "--"
    )

    plt.xlabel(
        "GIM-derived Ionospheric Correction (m)"
    )

    plt.ylabel(
        "Altimeter Filtered Ionospheric Correction (m)"
    )

    plt.title(
        "Sentinel-6 Altimeter vs GIM Ionospheric Correction"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 26. 图3：Altimeter - GIM correction时间序列
# ============================================================

if len(compare) > 1:

    plt.figure(
        figsize=(14, 5)
    )

    plt.scatter(
        compare["time"],
        compare["delta_iono_m"],
        s=8
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("UTC Time")
    plt.ylabel(
        "Altimeter filtered - GIM correction (m)"
    )

    plt.title(
        "Sentinel-6 Ionospheric Correction Residual"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 27. 图4：TEC vs GIM-derived correction
# ============================================================

if len(tmp_tec_gim) > 1:

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        tmp_tec_gim["TEC_TECU"],
        tmp_tec_gim["iono_cor_gim_m"],
        s=10,
        alpha=0.5
    )

    plt.xlabel("TEC (TECU)")
    plt.ylabel(
        "GIM-derived Ionospheric Correction (m)"
    )

    plt.title(
        "TEC vs GIM-derived Ionospheric Correction"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 28. 图5：TEC vs filtered Altimeter correction
# ============================================================

tmp_filtered = ocean[
    np.isfinite(
        ocean["iono_cor_alt_filtered_m"]
    )
].copy()

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    tmp_filtered["TEC_TECU"],
    tmp_filtered["iono_cor_alt_filtered_m"],
    s=10,
    alpha=0.5
)

plt.xlabel("TEC (TECU)")
plt.ylabel(
    "Filtered Altimeter Ionospheric Correction (m)"
)

plt.title(
    "TEC vs Filtered Altimeter Ionospheric Correction"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 29. 图6：连续TEC差分
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.scatter(
    diagnostic["time"],
    diagnostic["dTEC"],
    s=8
)

plt.xlabel("UTC Time")
plt.ylabel("|ΔTEC| (TECU)")
plt.title(
    "Sentinel-6 Adjacent TEC Difference"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 30. 图7：TEC按Rain Flag
# ============================================================

plt.figure(
    figsize=(14, 6)
)

for flag in sorted(
    ocean["rain_flag"]
    .dropna()
    .unique()
):

    subset = ocean[
        ocean["rain_flag"] == flag
    ]

    plt.scatter(
        subset["time"],
        subset["TEC_TECU"],
        s=10,
        label=f"rain_flag={int(flag)}"
    )

plt.xlabel("UTC Time")
plt.ylabel("TEC (TECU)")
plt.title(
    "Sentinel-6 TEC by Rain Flag"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 31. 导出 CSV
# ============================================================

if SAVE_CSV:

    stem = Path(
        file_path
    ).stem

    all_csv = (
        output_dir /
        f"{stem}_all_with_gim.csv"
    )

    ocean_csv = (
        output_dir /
        f"{stem}_ocean_with_gim.csv"
    )

    compare_csv = (
        output_dir /
        f"{stem}_alt_vs_gim.csv"
    )

    jumps_csv = (
        output_dir /
        f"{stem}_top_jumps.csv"
    )

    basic_qc_csv = (
        output_dir /
        f"{stem}_basic_qc.csv"
    )

    strict_qc_csv = (
        output_dir /
        f"{stem}_strict_qc_candidate.csv"
    )

    df.to_csv(
        all_csv,
        index=False
    )

    ocean.to_csv(
        ocean_csv,
        index=False
    )

    compare.to_csv(
        compare_csv,
        index=False
    )

    top_jumps.to_csv(
        jumps_csv,
        index=False
    )

    df_basic_qc.to_csv(
        basic_qc_csv,
        index=False
    )

    df_strict_candidate.to_csv(
        strict_qc_csv,
        index=False
    )

    print("\n" + "=" * 100)
    print("CSV输出完成")
    print("=" * 100)

    print(all_csv)
    print(ocean_csv)
    print(compare_csv)
    print(jumps_csv)
    print(basic_qc_csv)
    print(strict_qc_csv)


# ============================================================
# 32. 最终总结
# ============================================================

print("\n" + "=" * 100)
print("最终诊断总结")
print("=" * 100)

print(
    f"总记录数                  : {len(df)}"
)

print(
    f"Open Ocean有效TEC         : {len(ocean)}"
)

print(
    f"有效iono_cor_gim          : "
    f"{df['iono_cor_gim_m'].notna().sum()}"
)

print(
    f"Altimeter/GIM共同有效点   : {len(compare)}"
)

if len(compare) > 1:

    print(
        f"Altimeter filtered vs GIM R    : "
        f"{corr:.6f}"
    )

    print(
        f"Correction Bias                : "
        f"{bias:.6f} m"
    )

    print(
        f"Correction STD                 : "
        f"{std:.6f} m"
    )

    print(
        f"Correction RMSE                : "
        f"{rmse:.6f} m"
    )

    print(
        f"Correction MAE                 : "
        f"{mae:.6f} m"
    )

print(
    f"最大连续相邻TEC变化       : "
    f"{diagnostic['dTEC'].max():.3f} TECU"
)

print(
    f"TEC平均值                  : "
    f"{ocean['TEC_TECU'].mean():.3f} TECU"
)

print(
    f"TEC标准差                  : "
    f"{ocean['TEC_TECU'].std():.3f} TECU"
)

print(
    f"TEC最小值                  : "
    f"{ocean['TEC_TECU'].min():.3f} TECU"
)

print(
    f"TEC最大值                  : "
    f"{ocean['TEC_TECU'].max():.3f} TECU"
)

print("\n处理完成。")


# ============================================================
# 33. 关闭文件
# ============================================================

ds.close()
ds_ku.close()