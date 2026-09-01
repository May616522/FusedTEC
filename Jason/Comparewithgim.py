from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helper.interpolator import GIMInterpolator
# ============================================================
# 1. 批量处理 Jason CSV
# ============================================================

def process_jason_folder(
    folder_path,
    gim,
    compare_col="TEC_smooth",
    lat_res=2.5,
    lon_res=5.0,
    output_folder=None,
):
    """
    批量处理 Jason CSV 数据。

    功能
    ----
    1. 读取原始 Jason CSV
    2. 插值得到 GIM VTEC
    3. 计算 residual = gim_vtec - compare_col
    4. 将新数据保存到新目录，不覆盖原始 CSV
    5. 在线累计空间残差统计
    6. 在线累计每日残差统计
    7. 返回 spatial_df 和 daily_df

    Parameters
    ----------
    folder_path : str or Path
        原始 Jason CSV 文件夹。

    gim :
        GIMInterpolator 对象。

    compare_col : str
        Jason 中用于比较的 TEC 列，默认 "TEC_smooth"。

    lat_res : float
        纬度统计网格分辨率，单位 degree。

    lon_res : float
        经度统计网格分辨率，单位 degree。

    output_folder : str or Path or None
        保存插值后 CSV 的文件夹。
        如果为 None，则默认：
            folder_path / "with_gim"

    Returns
    -------
    spatial_df : pandas.DataFrame
        全球网格残差统计结果。

    daily_df : pandas.DataFrame
        每日残差统计结果。
    """

    # ========================================================
    # 路径
    # ========================================================

    folder_path = Path(folder_path)

    if output_folder is None:
        output_folder = folder_path / "with_gim"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # 只读取原始目录直接包含的 CSV
    # 不会读取 with_gim 子目录
    csv_files = sorted(
        folder_path.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"{folder_path} 中没有 CSV 文件"
        )

    print("=" * 60)
    print("Jason residual processing")
    print(f"Input folder : {folder_path}")
    print(f"Output folder: {output_folder}")
    print(f"CSV number   : {len(csv_files)}")
    print("=" * 60)

    # ========================================================
    # 在线统计容器
    #
    # value:
    #
    # [
    #     sum(residual),
    #     sum(residual^2),
    #     count
    # ]
    # ========================================================

    spatial_stats = defaultdict(
        lambda: [0.0, 0.0, 0]
    )

    daily_stats = defaultdict(
        lambda: [0.0, 0.0, 0]
    )

    success_count = 0
    failed_count = 0
    total_points = 0

    # ========================================================
    # 网格数量
    # ========================================================

    n_lat = int(
        np.ceil(180.0 / lat_res)
    )

    n_lon = int(
        np.ceil(360.0 / lon_res)
    )

    # ========================================================
    # 循环文件
    # ========================================================

    for i, csv_path in enumerate(
        csv_files,
        start=1
    ):

        print(
            f"[{i}/{len(csv_files)}] "
            f"Processing: {csv_path.name}"
        )

        try:
            df = pd.read_csv(csv_path)
            # ------------------------------------------------
            # 4. GIM 插值
            #
            # add_gim_to_dataframe 应该添加：
            #
            # df["gim_vtec"]
            #
            # ------------------------------------------------
            
            #    def add_gim_to_dataframe(df,gim,time_col="datetime",lat_col="lat",lon_col="lon",output_col="gim_vtec"):

            df = gim.add_gim_to_dataframe(
                df,
                time_col="datetime",
                lat_col="lat",
                lon_col="lon",
                output_col="gim_vtec")

            # ------------------------------------------------
            # 5. residual
            #
            # 正值：
            # GIM > Jason
            #
            # 负值：
            # GIM < Jason
            #
            # ------------------------------------------------

            df["residual"] = (df["gim_vtec"]- df[compare_col])

            # ------------------------------------------------
            # 6. 保存到新目录
            #
            # 原文件不覆盖
            # ------------------------------------------------

            output_path = (
                output_folder
                / csv_path.name
            )
            df.to_csv(
                output_path,
                index=False
            )

            # =================================================
            # 下面只用于统计
            #
            # 不需要复制全部列
            # =================================================

            stat_df = df[
                [
                    "datetime",
                    "lat",
                    "lon",
                    "residual"
                ]
            ].copy()

            # ------------------------------------------------
            # 去除异常值
            # ------------------------------------------------

            stat_df = stat_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            stat_df = stat_df.dropna(
                subset=[
                    "datetime",
                    "lat",
                    "lon",
                    "residual"
                ]
            )

            if stat_df.empty:

                print(
                    "    No valid residual data."
                )

                continue

            # ------------------------------------------------
            # 经纬度有效范围
            # ------------------------------------------------

            stat_df = stat_df[
                stat_df["lat"].between(
                    -90,
                    90
                )
                &
                stat_df["lon"].between(
                    -180,
                    180
                )
            ]

            if stat_df.empty:
                continue

            total_points += len(
                stat_df
            )

            # =================================================
            # 7. 每日日期
            # =================================================

            stat_df["date"] = (
                stat_df["datetime"]
                .dt.floor("D")
            )

            # =================================================
            # 8. 空间网格索引
            #
            # [-90,90]
            # [-180,180]
            # =================================================

            stat_df["lat_bin"] = np.floor(
                (
                    stat_df["lat"]
                    + 90.0
                )
                /
                lat_res
            ).astype(int)

            stat_df["lon_bin"] = np.floor(
                (
                    stat_df["lon"]
                    + 180.0
                )
                /
                lon_res
            ).astype(int)

            # 处理：
            #
            # lat = 90
            # lon = 180
            #
            # 导致索引越界的问题

            stat_df["lat_bin"] = (
                stat_df["lat_bin"]
                .clip(
                    0,
                    n_lat - 1
                )
            )

            stat_df["lon_bin"] = (
                stat_df["lon_bin"]
                .clip(
                    0,
                    n_lon - 1
                )
            )

            # =================================================
            # 9. 空间统计
            # =================================================

            grouped = stat_df.groupby(
                [
                    "lat_bin",
                    "lon_bin"
                ]
            )

            for key, group in grouped:

                r = (
                    group["residual"]
                    .to_numpy(
                        dtype=float
                    )
                )

                spatial_stats[key][0] += (
                    np.sum(r)
                )

                spatial_stats[key][1] += (
                    np.dot(r, r)
                )

                spatial_stats[key][2] += (
                    len(r)
                )

            # =================================================
            # 10. 每日统计
            # =================================================

            grouped_daily = stat_df.groupby(
                "date"
            )

            for date, group in grouped_daily:

                r = (
                    group["residual"]
                    .to_numpy(
                        dtype=float
                    )
                )

                daily_stats[date][0] += (
                    np.sum(r)
                )

                daily_stats[date][1] += (
                    np.dot(r, r)
                )

                daily_stats[date][2] += (
                    len(r)
                )

            success_count += 1

        except Exception as e:

            failed_count += 1

            print(
                f"    Failed: {csv_path.name}"
            )

            print(
                f"    Reason: {e}"
            )

    # ========================================================
    # 检查
    # ========================================================

    if success_count == 0:

        raise ValueError(
            "没有成功处理任何 Jason CSV"
        )

    # ========================================================
    # 11. 生成空间统计 DataFrame
    # ========================================================

    spatial_rows = []

    for (
        lat_idx,
        lon_idx
    ), (
        sum_r,
        sum_sq,
        count
    ) in spatial_stats.items():

        if count == 0:
            continue

        # ---------------------------------------------
        # Mean / Bias
        # ---------------------------------------------

        mean = (
            sum_r
            /
            count
        )

        # ---------------------------------------------
        # RMSE
        # ---------------------------------------------

        rmse = np.sqrt(
            sum_sq
            /
            count
        )

        # ---------------------------------------------
        # STD
        #
        # Var(X) = E(X²) - E(X)²
        # ---------------------------------------------

        variance = (
            sum_sq / count
            - mean ** 2
        )

        # 防止浮点数导致非常小的负数
        variance = max(
            variance,
            0.0
        )

        std = np.sqrt(
            variance
        )

        # ---------------------------------------------
        # 网格中心
        # ---------------------------------------------

        lat_center = (
            -90.0
            + lat_idx * lat_res
            + lat_res / 2.0
        )

        lon_center = (
            -180.0
            + lon_idx * lon_res
            + lon_res / 2.0
        )

        spatial_rows.append(
            {
                "lat": lat_center,
                "lon": lon_center,

                "mean": mean,
                "std": std,
                "rmse": rmse,

                "count": count
            }
        )

    spatial_df = pd.DataFrame(
        spatial_rows
    )

    if not spatial_df.empty:

        spatial_df = (
            spatial_df
            .sort_values(
                [
                    "lat",
                    "lon"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # 12. 生成每日统计 DataFrame
    # ========================================================

    daily_rows = []

    for date, (
        sum_r,
        sum_sq,
        count
    ) in daily_stats.items():

        if count == 0:
            continue

        mean = (
            sum_r
            /
            count
        )

        rmse = np.sqrt(
            sum_sq
            /
            count
        )

        variance = (
            sum_sq
            /
            count
            -
            mean ** 2
        )

        variance = max(
            variance,
            0.0
        )

        std = np.sqrt(
            variance
        )

        daily_rows.append(
            {
                "date": date,

                "mean": mean,
                "std": std,
                "rmse": rmse,

                "count": count
            }
        )

    daily_df = pd.DataFrame(
        daily_rows
    )

    if not daily_df.empty:

        daily_df = (
            daily_df
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # 13. 保存统计结果
    # ========================================================

    statistics_folder = (
        output_folder
        / "statistics"
    )

    statistics_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    spatial_path = (
        statistics_folder
        / "spatial_residual_statistics.csv"
    )

    daily_path = (
        statistics_folder
        / "daily_residual_statistics.csv"
    )

    spatial_df.to_csv(
        spatial_path,
        index=False
    )

    daily_df.to_csv(
        daily_path,
        index=False
    )

    # ========================================================
    # 输出处理信息
    # ========================================================

    print()
    print("=" * 60)
    print("Processing finished")
    print("=" * 60)

    print(
        f"Successful files : "
        f"{success_count}"
    )

    print(
        f"Failed files     : "
        f"{failed_count}"
    )

    print(
        f"Valid points     : "
        f"{total_points}"
    )

    print(
        f"Processed CSV    : "
        f"{output_folder}"
    )

    print(
        f"Spatial stats    : "
        f"{spatial_path}"
    )

    print(
        f"Daily stats      : "
        f"{daily_path}"
    )

    print("=" * 60)

    return spatial_df, daily_df


# ============================================================
# 2. 每日残差绘图
# ============================================================

def plot_daily_residual(
    daily_df,
    save_path=None
):
    """
    绘制每日平均残差以及 ±1 STD。
    """

    if daily_df.empty:
        raise ValueError(
            "daily_df 为空"
        )

    df = daily_df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    # 每日平均残差
    ax.plot(
        df["date"],
        df["mean"],
        label="Daily Mean Residual"
    )

    # ±1 STD
    ax.fill_between(
        df["date"],
        df["mean"] - df["std"],
        df["mean"] + df["std"],
        alpha=0.2,
        label="±1 STD"
    )

    # 零线
    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Residual (TECU)"
    )

    ax.set_title(
        "Daily GIM - Jason Residual"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.autofmt_xdate()

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 3. 每日 RMSE 绘图
# ============================================================

def plot_daily_rmse(
    daily_df,
    save_path=None
):
    """
    绘制每日 RMSE。
    """

    if daily_df.empty:
        raise ValueError(
            "daily_df 为空"
        )

    df = daily_df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["date"],
        df["rmse"]
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "RMSE (TECU)"
    )

    ax.set_title(
        "Daily GIM - Jason RMSE"
    )

    ax.grid(
        alpha=0.3
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 4. 全球空间平均残差
# ============================================================

def plot_spatial_mean(
    spatial_df,
    save_path=None,
    min_count=1
):
    """
    绘制全球空间平均残差。

    Parameters
    ----------
    min_count:
        一个网格至少包含多少数据点才显示。
    """

    if spatial_df.empty:
        raise ValueError(
            "spatial_df 为空"
        )

    df = spatial_df[
        spatial_df["count"]
        >= min_count
    ].copy()

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["mean"],
        s=20
    )

    cbar = plt.colorbar(
        sc,
        ax=ax
    )

    cbar.set_label(
        "Mean Residual (TECU)"
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.set_ylabel(
        "Latitude (°)"
    )

    ax.set_title(
        "Global Mean GIM - Jason Residual"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 5. 全球空间 RMSE
# ============================================================

def plot_spatial_rmse(
    spatial_df,
    save_path=None,
    min_count=1
):
    """
    绘制全球空间 RMSE。
    """

    if spatial_df.empty:
        raise ValueError(
            "spatial_df 为空"
        )

    df = spatial_df[
        spatial_df["count"]
        >= min_count
    ].copy()

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["rmse"],
        s=20
    )

    cbar = plt.colorbar(
        sc,
        ax=ax
    )

    cbar.set_label(
        "RMSE (TECU)"
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.set_ylabel(
        "Latitude (°)"
    )

    ax.set_title(
        "Global GIM - Jason RMSE"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 6. 全球样本数量
# ============================================================

def plot_spatial_count(
    spatial_df,
    save_path=None
):
    """
    绘制每个空间网格中的 Jason 样本数量。
    """

    if spatial_df.empty:
        raise ValueError(
            "spatial_df 为空"
        )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    sc = ax.scatter(
        spatial_df["lon"],
        spatial_df["lat"],
        c=spatial_df["count"],
        s=20
    )

    cbar = plt.colorbar(
        sc,
        ax=ax
    )

    cbar.set_label(
        "Sample Count"
    )

    ax.set_xlim(
        -180,
        180
    )

    ax.set_ylim(
        -90,
        90
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.set_ylabel(
        "Latitude (°)"
    )

    ax.set_title(
        "Jason Sample Distribution"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# ============================================================
# 7. 主程序示例
# ============================================================

if __name__ == "__main__":


    folder_path = r"F:\FusedTec\Data\Jason2020\test1"
    output_folder = r"F:\FusedTec\Data\Jason2020\TecCSV1"


    gim = GIMInterpolator(r"F:\FusedTec\Data\GIM2020\COD")
    spatial_df, daily_df = process_jason_folder(
        folder_path=folder_path,
        gim=gim,
        compare_col="TEC_smooth",
        lat_res=2.5,
        lon_res=5.0,
        output_folder=output_folder
    )

    figure_folder = (
        Path(output_folder)
        / "figures"
    )

    plot_daily_residual(
        daily_df,
        figure_folder
        / "daily_residual.png"
    )

    plot_daily_rmse(
        daily_df,
        figure_folder
        / "daily_rmse.png"
    )

    plot_spatial_mean(
        spatial_df,
        figure_folder
        / "global_mean_residual.png",
        min_count=20
    )

    plot_spatial_rmse(
        spatial_df,
        figure_folder
        / "global_rmse.png",
        min_count=20
    )

    plot_spatial_count(
        spatial_df,
        figure_folder
        / "global_sample_count.png"
    )
    
    