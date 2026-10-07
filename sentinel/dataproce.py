import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 文件路径
# ============================================================

file_path = (
    r"F:\FusedTec\Data\Sentinel2021"
    r"\S6A_P4_2__LR_RED__NT_041_188_20211226T142429_20211226T152042_G01.nc"
)


# ============================================================
# 2. 常数
# ============================================================

F_KU = 13.575e9       # Hz
K_IONO = 40.308
TECU = 1e16

TEC_FRACTION = 0.881


# ============================================================
# 3. 读取 data_01
# ============================================================

ds = xr.open_dataset(
    file_path,
    group="data_01"
)


# ============================================================
# 4. 读取变量
# ============================================================

time = ds["time"].values
lat = ds["latitude"].values
lon = ds["longitude"].values

surface_type = (
    ds["surface_classification_flag"].values
)

iono_filtered = (
    ds["iono_cor_alt_filtered"].values
)

tec_official = (
    ds["total_electron_content"].values
    / TECU
)


# ============================================================
# 5. 由 filtered correction 反算轨道以下TEC
# ============================================================

TEC_below = (
    -iono_filtered
    * F_KU**2
    /
    (
        K_IONO
        * TECU
    )
)


# ============================================================
# 6. 做0.881 topside correction
# ============================================================

TEC_total_0881 = (
    TEC_below
    / TEC_FRACTION
)


# ============================================================
# 7. DataFrame
# ============================================================

df = pd.DataFrame({
    "time": time,
    "lat": lat,
    "lon": lon,

    "TEC_below": TEC_below,
    "TEC_total_0881": TEC_total_0881,
    "TEC_official": tec_official,

    "surface_type": surface_type
})


# ============================================================
# 8. 只保留open ocean + 三者共同有效
# ============================================================

compare = df[
    (df["surface_type"] == 0)
    &
    np.isfinite(df["TEC_below"])
    &
    np.isfinite(df["TEC_total_0881"])
    &
    np.isfinite(df["TEC_official"])
].copy()


print("=" * 100)
print("有效比较点数")
print("=" * 100)

print(len(compare))


# ============================================================
# 9. 指标函数
# ============================================================

def calc_metrics(x, y, name):

    x = np.asarray(x)
    y = np.asarray(y)

    # Pearson R
    r = np.corrcoef(x, y)[0, 1]

    # y = slope*x + intercept
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    residual = y - x

    bias = np.mean(residual)

    std = np.std(
        residual,
        ddof=1
    )

    rmse = np.sqrt(
        np.mean(
            residual**2
        )
    )

    mae = np.mean(
        np.abs(
            residual
        )
    )

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print(
        f"R         = {r:.6f}"
    )

    print(
        f"Slope     = {slope:.6f}"
    )

    print(
        f"Intercept = {intercept:.6f} TECU"
    )

    print(
        f"Bias      = {bias:.6f} TECU"
    )

    print(
        f"STD       = {std:.6f} TECU"
    )

    print(
        f"RMSE      = {rmse:.6f} TECU"
    )

    print(
        f"MAE       = {mae:.6f} TECU"
    )

    return {
        "R": r,
        "Slope": slope,
        "Intercept": intercept,
        "Bias": bias,
        "STD": std,
        "RMSE": rmse,
        "MAE": mae
    }


# ============================================================
# 10. 比较1：below vs official
# ============================================================

metrics_below = calc_metrics(
    compare["TEC_below"],
    compare["TEC_official"],
    "TEC_below vs TEC_official"
)


# ============================================================
# 11. 比较2：0.881 corrected vs official
# ============================================================

metrics_total = calc_metrics(
    compare["TEC_total_0881"],
    compare["TEC_official"],
    "TEC_total_0881 vs TEC_official"
)


# ============================================================
# 12. 计算比值
# ============================================================

compare["ratio_official_below"] = (
    compare["TEC_official"]
    / compare["TEC_below"]
)

compare["ratio_official_total"] = (
    compare["TEC_official"]
    / compare["TEC_total_0881"]
)


print("\n" + "=" * 100)
print("比值统计")
print("=" * 100)

print("\nOfficial / Below:")

print(
    compare[
        "ratio_official_below"
    ].describe()
)

print(
    "\n理论 1/0.881 = "
    f"{1 / TEC_FRACTION:.6f}"
)


print("\nOfficial / Total_0881:")

print(
    compare[
        "ratio_official_total"
    ].describe()
)


# ============================================================
# 13. 图1：Below vs Official
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    compare["TEC_below"],
    compare["TEC_official"],
    s=10,
    alpha=0.5
)

vmin = min(
    compare["TEC_below"].min(),
    compare["TEC_official"].min()
)

vmax = max(
    compare["TEC_below"].max(),
    compare["TEC_official"].max()
)

plt.plot(
    [vmin, vmax],
    [vmin, vmax],
    "--",
    label="1:1"
)

plt.xlabel(
    "TEC below orbit from filtered correction (TECU)"
)

plt.ylabel(
    "Official total_electron_content (TECU)"
)

plt.title(
    f"Below-orbit TEC vs Official TEC\n"
    f"R={metrics_below['R']:.4f}, "
    f"Slope={metrics_below['Slope']:.4f}"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 14. 图2：Topside corrected vs Official
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    compare["TEC_total_0881"],
    compare["TEC_official"],
    s=10,
    alpha=0.5
)

vmin = min(
    compare["TEC_total_0881"].min(),
    compare["TEC_official"].min()
)

vmax = max(
    compare["TEC_total_0881"].max(),
    compare["TEC_official"].max()
)

plt.plot(
    [vmin, vmax],
    [vmin, vmax],
    "--",
    label="1:1"
)

plt.xlabel(
    "TEC after 0.881 topside correction (TECU)"
)

plt.ylabel(
    "Official total_electron_content (TECU)"
)

plt.title(
    f"Topside-corrected TEC vs Official TEC\n"
    f"R={metrics_total['R']:.4f}, "
    f"Slope={metrics_total['Slope']:.4f}"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 15. 图3：三者时间序列
# ============================================================

plt.figure(
    figsize=(15, 6)
)

plt.plot(
    compare["time"],
    compare["TEC_below"],
    label="TEC below orbit",
    linewidth=1
)

plt.plot(
    compare["time"],
    compare["TEC_total_0881"],
    label="TEC after /0.881",
    linewidth=1
)

plt.plot(
    compare["time"],
    compare["TEC_official"],
    label="Official TEC",
    linewidth=1.5
)

plt.xlabel("UTC Time")
plt.ylabel("TEC (TECU)")

plt.title(
    "Topside Correction Verification"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 16. 图4：ratio时间序列
# ============================================================

plt.figure(
    figsize=(15, 5)
)

plt.scatter(
    compare["time"],
    compare["ratio_official_below"],
    s=8,
    label="Official / Below"
)

plt.axhline(
    1 / TEC_FRACTION,
    linestyle="--",
    linewidth=1,
    label=f"1 / 0.881 = {1 / TEC_FRACTION:.4f}"
)

plt.xlabel("UTC Time")

plt.ylabel(
    "TEC Ratio"
)

plt.title(
    "Official TEC / Below-orbit TEC"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 17. 最终判断提示
# ============================================================

print("\n" + "=" * 100)
print("判断参考")
print("=" * 100)

print(
    "如果 Official / Below 的均值接近 "
    f"{1 / TEC_FRACTION:.4f}，"
)

print(
    "并且 TEC_total_0881 vs Official 的 "
    "Slope≈1、Bias≈0、RMSE显著更小，"
)

print(
    "则说明 official total_electron_content "
    "确实应用了约0.881的topside correction。"
)


ds.close()