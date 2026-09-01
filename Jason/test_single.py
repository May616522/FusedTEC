import os
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from jason_processing import (
    process_jason,
    JasonConfig,
    load_landmask
)


# ======================
# 单文件路径
# ======================

nc_file = r"F:\FusedTec\Data\Jason2020\JA3_GPN_2PfP180_135_20210101_133741_20210101_143354.nc"



# ======================
# 参数
# ======================

cfg = JasonConfig()
mask_ds=load_landmask(cfg.mask_path)


# ======================
# 处理
# ======================

df = process_jason(
    nc_file,
    mask_ds,
    cfg
)


def plot_tec_map(df, value="TEC_smooth"):

    """
    绘制Jason TEC空间分布

    value:
        TEC_raw
        TEC_smooth
    """


    fig = plt.figure(
        figsize=(12,6)
    )


    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )


    # 设置地图范围
    ax.set_extent(
        [
            df.lon.min()-5,
            df.lon.max()+5,
            df.lat.min()-5,
            df.lat.max()+5
        ],
        crs=ccrs.PlateCarree()
    )


    # 添加地理要素

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8
    )


    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.5
    )


    ax.gridlines(
        draw_labels=True
    )



    # TEC散点

    sc=ax.scatter(
        df.lon,
        df.lat,
        c=df[value],
        s=8,
        cmap="jet",
        transform=ccrs.PlateCarree()
    )


    # colorbar

    cbar=plt.colorbar(
        sc,
        ax=ax,
        orientation="vertical",
        pad=0.02
    )


    cbar.set_label(
        "TEC (TECU)"
    )


    plt.title(
        f"Jason {value} spatial distribution"
    )


    plt.show()

print("="*50)

print("处理完成")

print(df.head())

print(df.info())


print("="*50)

print(
    "时间范围:"
)

print(
    df.datetime.min(),
    "---->",
    df.datetime.max()
)


print(
    "纬度范围:"
)

print(
    df.lat.min(),
    df.lat.max()
)


print(
    "经度范围:"
)

print(
    df.lon.min(),
    df.lon.max()
)


print(
    "TEC统计:"
)

print(
    df[
        [
        "TEC_raw",
        "TEC_smooth"
        ]
    ]
    .describe()
)

plot_tec_map(
    df,
    value="TEC_smooth"
)



