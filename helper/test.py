import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from interpolator import GIMInterpolator



def analyze_vtec(df, compare_col, gim_col="gim_vtec"):
    """
    分析 GIM VTEC 与另一列 VTEC 的相关性和残差。
        correlation : Pearson相关系数
        bias        : 平均残差
        std         : 残差标准差
        rmse        : 均方根误差
        mae         : 平均绝对误差
        n           : 有效样本数
    """

    data = df[[gim_col, compare_col]].copy()


    if "time" in df.columns:    #保留时间列
        data["time"] = pd.to_datetime(df["time"])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[gim_col, compare_col])
    if len(data) == 0:
        raise ValueError("没有有效数据可以进行分析")
    gim = data[gim_col].to_numpy()
    obs = data[compare_col].to_numpy()
    residual = gim - obs
    data["residual"] = residual

    # 3. 统计指标
    correlation = np.corrcoef(gim, obs)[0, 1]
    bias = np.mean(residual)
    std = np.std(residual, ddof=1)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))

    stats = {
        "correlation": correlation,
        "bias": bias,
        "std": std,
        "rmse": rmse,
        "mae": mae,
        "n": len(data)
    }


    # 4. 散点图
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(obs,gim,s=8,alpha=0.5)
    # y = x 参考线
    vmin = min(obs.min(), gim.min())
    vmax = max(obs.max(), gim.max())

    ax.plot(
        [vmin, vmax],
        [vmin, vmax],
        "--",
        linewidth=1.5,
        label="1:1 line"
    )

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel(f"{compare_col} (TECU)")
    ax.set_ylabel(f"{gim_col} (TECU)")
    ax.set_title("VTEC comparison")

    # 显示统计指标
    text = (
        f"N = {len(data)}\n"
        f"R = {correlation:.3f}\n"
        f"Bias = {bias:.3f} TECU\n"
        f"STD = {std:.3f} TECU\n"
        f"RMSE = {rmse:.3f} TECU\n"
        f"MAE = {mae:.3f} TECU"
    )

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.8)
    )

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # =========================
    # 5. 残差随时间变化
    # =========================
    if "time" in data.columns:

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.scatter(
            data["time"],
            residual,
            s=6,
            alpha=0.5
        )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        # Bias 线
        ax.axhline(
            bias,
            linestyle="-",
            linewidth=1.5,
            label=f"Bias = {bias:.2f} TECU"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"{gim_col} - {compare_col} (TECU)")
        ax.set_title("VTEC residual versus time")

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    # =========================
    # 6. 残差直方图
    # =========================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        residual,
        bins=50,
        alpha=0.7
    )

    ax.axvline(
        bias,
        linestyle="--",
        linewidth=1.5,
        label=f"Bias = {bias:.2f}"
    )

    ax.axvline(
        bias - std,
        linestyle=":",
        linewidth=1.2,
        label=f"±1 STD = {std:.2f}"
    )

    ax.axvline(
        bias + std,
        linestyle=":",
        linewidth=1.2
    )

    ax.set_xlabel(f"{gim_col} - {compare_col} (TECU)")
    ax.set_ylabel("Count")
    ax.set_title("VTEC residual distribution")

    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # =========================
    # 7. 打印结果
    # =========================
    print("=" * 40)
    print(f"GIM column      : {gim_col}")
    print(f"Compare column  : {compare_col}")
    print(f"N               : {len(data)}")
    print(f"Correlation R   : {correlation:.4f}")
    print(f"Bias            : {bias:.4f} TECU")
    print(f"STD             : {std:.4f} TECU")
    print(f"RMSE            : {rmse:.4f} TECU")
    print(f"MAE             : {mae:.4f} TECU")
    print("=" * 40)

    return stats

gim = GIMInterpolator(
    r"F:\FusedTec\Data\GIM2020\COD")
df = gim.add_gim_to_dataframe(
    r"F:\FusedTec\Data\Jason2020\TecCSV\Jason_20200222T155753to20200222T164944.csv",gim)

analyze_vtec(df, "TEC_smooth", gim_col="gim_vtec")




