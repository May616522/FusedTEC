"""
Jason altimetry TEC processing

pipeline:

nc file
 |
 |-- read
 |
 |-- qc
 |
 |-- calculate TEC
 |
 |-- export

"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import glob

from dataclasses import dataclass

# 参数
@dataclass
class JasonConfig:

    freq_ku: float = 13.575e9
    k_const: float = 40.3
    tecu_scale: float = 1e16
    smooth_window: str = "16s"
    iono_min: float = -0.4
    iono_max: float = 0.04
    range_num_min: int = 10
    range_rms_max: float = 0.2
    
    mask_path: str = r"F:\bs\landmask_2021_10_12\landmask_static.nc"

# 1.读取文件
def read_jason_nc(file_path):
    """
    读取Jason nc文件 return:  dataframe
    """

    with xr.open_dataset(file_path,group="data_01",decode_times=False) as ds:
        df = ds[
            [
                "time",
                "latitude",
                "longitude",
                "altitude",
                "surface_classification_flag",
                "ice_flag",
                "rain_flag"
            ]
        ].to_dataframe().reset_index()
        
    df=df.rename(columns={
            "latitude":"lat",
            "longitude":"lon",
            "altitude":"alt",
            "surface_classification_flag":"surface",
            "ice_flag":"ice",
            "rain_flag":"rain"}
    )

    with xr.open_dataset(file_path,group="data_01/ku",decode_times=False) as ds:
        df["iono"]=(
            ds["iono_cor_alt_filtered"].values)
        df["range_numval"]=(ds["range_ocean_numval"].values)
        df["range_rms"]=(ds["range_ocean_rms"].values)

    return df

# 2.时间解析,存疑，后期修改
def parse_time(df):
    epoch=pd.Timestamp("2000-01-01")
    df["datetime"]=( epoch+pd.to_timedelta(df.time,unit="s"))
    return df

#导入陆地海洋静态掩码
def load_landmask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    print("loading land mask...")
    ds=xr.open_dataset(mask_path).load()

    # 经度统一
    ds["longitude"]=np.mod(ds["longitude"],360)
    ds=ds.sortby(["latitude","longitude"])
    return ds

def add_land_mask(df,mask_ds):
    """
    给Jason点添加海陆属性
    lsm:0   ocean 1   land
    """
    lat=xr.DataArray(df["lat"].values, dims="points")
    lon=xr.DataArray(df["lon"].values%360,dims="points")
    df["land_mask"]=( mask_ds["lsm"].sel(latitude=lat,longitude=lon,method="nearest").values)
    return df

# 3.质量控制
def quality_control(df,cfg,mask_ds):
    #经过实验发现这一步剔除最多，从3100直接到1700
    mask=(df.surface==0) #只取海洋，对于冰和雨另作考虑
    mask &= (df.range_numval>=cfg.range_num_min)
    mask &= (df.range_rms<=cfg.range_rms_max)
    mask &= (df.iono>=cfg.iono_min)
    mask &= (df.iono<=cfg.iono_max)
    
    df= df[mask].copy()

    #这是否要加一个海陆边界控制(已添加)
    df=add_land_mask(df,mask_ds)
    df=df[df.land_mask <0.5].copy()
    save_cols=["datetime","lat","lon","iono","alt","ice","rain"]
    df=df[save_cols]
    return df



# 4.TEC计算
def calculate_TEC(df,cfg):

    df=df.sort_values("datetime")#按道理这里读取的时候就是排好的
    df["iono_smooth"]=(df.rolling(
            cfg.smooth_window,
            on="datetime",
            center=True
        )
        ["iono"]
        .mean()
    )#窗口滑动平均，但是我的疑惑是，数据本身就是平均之后的

    factor=(cfg.freq_ku**2/(cfg.k_const*cfg.tecu_scale))
    df["TEC_raw"]=(abs(np.minimum(df.iono,0))*factor)
    df["TEC_smooth"]=(abs(np.minimum( df.iono_smooth, 0))* factor)
    df["alt"] /=1000 #单位换算
    
    #去掉零值
    df = df[(df["TEC_raw"] != 0) & (df["TEC_smooth"] != 0)]
    return df

def get_nc_files(input_dir):
    """
    获取文件夹下所有nc文件
    """
    nc_files = glob.glob(os.path.join(input_dir,"*.nc"))
    if len(nc_files)==0:raise FileNotFoundError("没有找到nc文件")
    return sorted(nc_files)

def process_jason(file_path,mask_ds,cfg=None):

    if cfg is None:
        cfg=JasonConfig()

    df=read_jason_nc(file_path)
    print("orignal data nums:%d\n",len(df))
    df=parse_time(df)
    df=quality_control(df,cfg,mask_ds)
    print("QC data nums:%d\n",len(df))
    df=calculate_TEC(df,cfg)
    print("cal data nums:%d\n",len(df))
    return df

def batch_process_jason(input_dir,output_dir):
    """
    批量处理Jason nc文件
    """
    nc_files = get_nc_files(input_dir)
    os.makedirs(output_dir,exist_ok=True)
    total=len(nc_files)
    success=0
    
    #导入陆地海洋静态掩码
    cfg=JasonConfig()
    mask_ds=load_landmask(cfg.mask_path)

    
    for i,file in enumerate(nc_files,1):
        name=os.path.basename(file)
        try:
            print(f"[{i}/{total}] processing {name}")
            df=process_jason(file,mask_ds)
            if df.empty:
                continue
            
            time_start = (df["datetime"].min())
            time_end = (df["datetime"].max())
            t1=time_start.strftime("%Y%m%dT%H%M%S")
            t2=time_end.strftime("%Y%m%dT%H%M%S")
            
            out_name = ( f"Jason_{t1}to{t2}.csv")
            out_path=os.path.join(output_dir,out_name)

            save_cols=["datetime","lat","lon","TEC_raw","TEC_smooth","alt","ice","rain"]
            df[save_cols].to_csv(out_path,index=False)
            
            success+=1
            print(f"saved: {out_name}")

        except Exception as e:
            print(f"{name} failed:")
            print(e)

    print("="*50)
    print(f"总文件:{total}")
    print(f"成功:{success}")
    
    
if __name__ == "__main__":


    input_dir = r"F:\FusedTec\Data\Jason2020"
    output_dir = r"F:\FusedTec\Data\Jason2020\TecCSV"
    batch_process_jason(
        input_dir,
        output_dir
    )

