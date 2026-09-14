from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# Sentinel-3 RED product -> TEC
# ============================================================

# Sentinel-3 SRAL Ku-band frequency
FREQ_KU = 13.575e9      # Hz

# First-order ionospheric constant
K_IONO = 40.3

# 1 TECU = 1e16 electrons / m^2
TECU_SCALE = 1e16

# Ku-band range correction (m) -> TECU conversion factor.
TEC_PER_METER_KU = FREQ_KU**2 / K_IONO / TECU_SCALE


def iono_correction_to_tec(iono_cor_m):
    """
    将 Sentinel-3 Ku 波段电离层距离改正转换为 TECU。

    Sentinel-3 中 iono_cor_alt_* 的定义是：
    为修正电离层延迟，需要加到 instrument range 上的 correction。
    因此通常为负值。

    电离层群延迟：
        delay = 40.3 * TEC / f^2

    Sentinel 产品给的是 correction = -delay，因此：

        TEC = -correction * f^2 / 40.3

    Parameters
    ----------
    iono_cor_m : array-like
        电离层距离改正，单位 m。

    Returns
    -------
    tec : array-like
        TEC，单位 TECU。
    """
    iono_cor_m = np.asarray(iono_cor_m, dtype=float)

    tec = -iono_cor_m * TEC_PER_METER_KU

    return tec


def read_sentinel3_red(
    nc_path,
    use_filtered=True,
    ocean_only=True,
    tec_min=0.0,
    tec_max=200.0,
):
    """
    读取 Sentinel-3 SR_2_WAT RED 产品，并转换得到 1 Hz VTEC。

    ``iono_cor_alt_*`` 只对开阔海洋有效。因此 ``ocean_only=True``
    时会同时要求 ``surf_type_01 == 0`` 和
    ``open_sea_ice_flag_01_ku == 0``，避免把海冰、混合表面和无法
    分类的高纬数据误当作海洋观测。

    Parameters
    ----------
    nc_path : str or Path
        Sentinel-3 RED NetCDF 文件路径。

    use_filtered : bool
        True:
            使用 iono_cor_alt_filtered_01_ku
        False:
            使用 iono_cor_alt_01_ku

    ocean_only : bool
        是否仅保留纯海洋数据。

        surf_type_01:
            0 = ocean_or_semi_enclosed_sea
            1 = enclosed_sea_or_lake
            2 = continental_ice
            3 = land

        open_sea_ice_flag_01_ku:
            0 = ocean
            1~5 = 海冰、混合表面或无法分类，应从双频 VTEC 中排除

    tec_min : float
        TEC 最小允许值，单位 TECU。

    tec_max : float
        TEC 最大允许值，单位 TECU。

    Returns
    -------
    df : pandas.DataFrame
        包含：
            time
            lat
            lon
            iono_cor
            tec
            surf_type
            open_sea_ice_flag
    """

    nc_path = Path(nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"文件不存在: {nc_path}")

    # xarray 默认会自动处理 scale_factor 和 add_offset
    # 同时通常会将 _FillValue 转换为 NaN
    with xr.open_dataset(nc_path) as ds:

        # ----------------------------------------------------
        # 1. 选择电离层改正变量
        # ----------------------------------------------------
        if use_filtered:
            iono_var = "iono_cor_alt_filtered_01_ku"
        else:
            iono_var = "iono_cor_alt_01_ku"

        required_vars = [
            "time_01",
            "lat_01",
            "lon_01",
            "surf_type_01",
            iono_var,
        ]
        if ocean_only:
            required_vars.append("open_sea_ice_flag_01_ku")

        missing_vars = [
            var for var in required_vars
            if var not in ds.variables
        ]

        if missing_vars:
            raise KeyError(
                f"文件缺少变量: {missing_vars}"
            )

        # ----------------------------------------------------
        # 2. 读取数据
        # ----------------------------------------------------
        time = ds["time_01"].values
        lat = ds["lat_01"].values.astype(float)
        lon = ds["lon_01"].values.astype(float)

        surf_type = ds["surf_type_01"].values
        iono_cor = ds[iono_var].values.astype(float)
        if "open_sea_ice_flag_01_ku" in ds.variables:
            open_sea_ice_flag = ds["open_sea_ice_flag_01_ku"].values
        else:
            # 允许 ocean_only=False 时读取不含此标志的旧产品，但在输出中
            # 明确记为缺失，不能误解为纯海洋。
            open_sea_ice_flag = np.full(iono_cor.shape, np.nan)

    # --------------------------------------------------------
    # 3. 时间转换
    #
    # time_01:
    # seconds since 2000-01-01 00:00:00
    # --------------------------------------------------------
    time = pd.to_datetime(time)

    # --------------------------------------------------------
    # 4. 经度统一到 [-180, 180]
    #
    # Sentinel-3 有些轨道经度可能使用 0~360
    # --------------------------------------------------------
    lon = np.where(
        lon > 180.0,
        lon - 360.0,
        lon
    )

    # --------------------------------------------------------
    # 5. 电离层改正 -> TEC
    # --------------------------------------------------------
    tec = iono_correction_to_tec(iono_cor)

    # --------------------------------------------------------
    # 6. 构建 DataFrame
    # --------------------------------------------------------
    df = pd.DataFrame({
        "datetime": time,
        "lat": lat,
        "lon": lon,
        "surf_type": surf_type,
        "open_sea_ice_flag": open_sea_ice_flag,
        "iono_cor": iono_cor,
        "tec": tec,
    })

    input_count = len(df)

    # --------------------------------------------------------
    # 7. 基础 QC
    # --------------------------------------------------------

    # 删除 NaN / inf
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "datetime",
            "lat",
            "lon",
            "iono_cor",
            "tec",
        ]
    )

    # --------------------------------------------------------
    # 8. 纯海洋筛选
    #
    # surf_type_01 == 0 并不能排除高纬海冰。对于 altimeter-derived
    # ionospheric correction，还必须使用 open_sea_ice_flag 进行筛选。
    # --------------------------------------------------------
    if ocean_only:
        df = df[
            (df["surf_type"] == 0)
            & (df["open_sea_ice_flag"] == 0)
        ].copy()

    pure_ocean_count = len(df)

    # --------------------------------------------------------
    # 9. TEC 合理范围筛选
    # --------------------------------------------------------
    df = df[
        (df["tec"] >= tec_min)
        &
        (df["tec"] <= tec_max)
    ].copy()

    # --------------------------------------------------------
    # 10. 排序
    # --------------------------------------------------------
    df = df.sort_values("datetime")
    df = df.reset_index(drop=True)

    df.attrs["qc_counts"] = {
        "input": input_count,
        "after_surface_filter": pure_ocean_count,
        "output": len(df),
    }

    return df


# ============================================================
# 例子
# ============================================================

if __name__ == "__main__":

    nc_path = (
        r"F:\FusedTec\Data\Sentinel2021\S3B_SR_2_WAT____20210609T191402_20210609T200209_20260315T212241_2886_053_213__426_MAR_R_NT_G62.SEN3\S3B_SR_2_WAT____20210609T191402_20210609T200209_20260315T212241_2886_053_213__426_MAR_R_NT_G62.SEN3\S3B_SR_2_WAT_RED__NT_053_426_20210609T191402_20210609T200209_G62.nc"
    )

    df = read_sentinel3_red(
        nc_path,
        use_filtered=True,
        ocean_only=True,
        tec_min=0,
        tec_max=200,
    )

    print(df.head())

    print("\n============================")
    print("Sentinel-3 TEC statistics")
    print("============================")

    qc_counts = df.attrs.get("qc_counts", {})
    print(f"Input records          : {qc_counts.get('input', 'N/A')}")
    print(
        "After ocean/ice filter : "
        f"{qc_counts.get('after_surface_filter', 'N/A')}"
    )

    print(f"Number of valid points : {len(df)}")
    print(f"TEC mean               : {df['tec'].mean():.3f} TECU")
    print(f"TEC std                : {df['tec'].std():.3f} TECU")
    print(f"TEC min                : {df['tec'].min():.3f} TECU")
    print(f"TEC max                : {df['tec'].max():.3f} TECU")

    # 保存
    output_path = Path(
        r"F:\FusedTec\Data\Sentinel2021\sentinel3_tec.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved to: {output_path}")
