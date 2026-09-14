from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from .GIM_Read import parse_ionex_grid

class GIMInterpolator:

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # 缓存已经读取过的 GIM
        self.cache = {}

    def find_file(self, target_time):
        """
        根据日期找到对应 GIM 文件。
        """
        doy = target_time.strftime("%j")
        year_simple = target_time.strftime("%y")
        gim_dir = self.root_dir / doy
        if not gim_dir.exists():
            return None
        files = list(
            gim_dir.glob(
                f"*{doy}*.{year_simple}i*"
            )
        )
        if not files:
            return None
        return files[0]

    def load_day(self, target_time):
        """
        读取某一天 GIM。
        如果已经读取过，则直接从 cache 返回。
        """
        date_key = target_time.strftime("%Y%m%d")
        if date_key in self.cache:
            return self.cache[date_key]
        file_path = self.find_file(target_time)
        if file_path is None:
            raise FileNotFoundError(
                f"找不到 {target_time.date()} 的 GIM 文件")
        times, lats, lons, tec = parse_ionex_grid(
            file_path)
        data = {
            "times": times,
            "lats": lats,
            "lons": lons,
            "tec": tec,}
        self.cache[date_key] = data
        return data
    
    def query(self, target_time, lat, lon):
        """
        查询一个时空位置的 GIM TEC,空间双线性插值,时间线性插值
        """

        # IONEX epochs are parsed as timezone-naive UTC datetimes.  Inputs can
        # be either Jason timestamps without a timezone or ISO-8601 timestamps
        # from COSMIC (for example, ``2021-02-24T23:59:52Z``).  Convert both to
        # the same representation before file lookup and datetime arithmetic.
        target_time = self._normalize_time(target_time)

        data = self.load_day(target_time)
        times = data["times"]
        lats = data["lats"]
        lons = data["lons"]
        tec = data["tec"]

        lon = self._normalize_lon(lon,lons)
        times_np = np.array(times,dtype="datetime64[s]")
        target_np = np.datetime64(target_time,"s")
        idx = np.searchsorted(times_np, target_np)#前面有几个数据

        # 正好等于某一 TEC MAP
        if (idx < len(times_np)and times_np[idx] == target_np):
            return self._spatial_interp(lats,lons,tec[idx],lat,lon)

        # 时间边界
        if idx == 0:
            raise ValueError(
                f"{target_time} 早于第一张 TEC MAP")
        if idx >= len(times):
            raise ValueError(
                f"{target_time} 晚于最后一张 TEC MAP")

        i1 = idx - 1
        i2 = idx
        t1 = times[i1]
        t2 = times[i2]


        # 两个时间层分别空间插值
        v1 = self._spatial_interp(lats,lons,tec[i1],lat,lon)
        v2 = self._spatial_interp(lats,lons,tec[i2],lat,lon)

        # 时间线性插值
        alpha = ((target_time - t1).total_seconds()/(t2 - t1).total_seconds())
        value = (v1 + alpha * (v2 - v1))
        return float(value)
    

    @staticmethod
    def _spatial_interp(lats,lons,grid,lat,lon):
        """
        对单个 TEC MAP 做二维双线性插值。
        """

        # 要求坐标严格递增
        if lats[0] > lats[-1]:
            lats_use = lats[::-1]
            grid_use = grid[::-1, :]
        else:
            lats_use = lats
            grid_use = grid

        if lons[0] > lons[-1]:
            lons_use = lons[::-1]
            grid_use = grid_use[:, ::-1]
        else:
            lons_use = lons

        interpolator = RegularGridInterpolator((lats_use,lons_use),
            grid_use,
            method="linear",
            bounds_error=False,
            fill_value=np.nan)

        value = interpolator([[lat, lon]])[0]
        return float(value)
    
    @staticmethod
    def _normalize_lon(lon, gim_lons):
        """
        将输入经度转换到 GIM 使用的范围。-180-180 -90-90
        """
        "对于jaosn输入数据,精度在0-360，需要进行处理"
        lon = (float(lon) + 180.0)% 360.0 - 180.0
        return lon

    @staticmethod
    def _normalize_time(value):
        
        """Return a timezone-naive UTC datetime for GIM interpolation."""

        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError(f"无效时间: {value!r}")
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.to_pydatetime(warn=False)
    
    
    def add_gim_to_dataframe(self,df,time_col="datetime",lat_col="lat",lon_col="lon",output_col="gim_vtec"):
        """
        给任意 DataFrame 添加 GIM VTEC。
        """

        df = df.copy()
        # utc=True accepts both Jason's timezone-naive timestamps and COSMIC's
        # trailing-Z timestamps.  Removing the timezone afterwards matches the
        # timezone-naive UTC epochs read from IONEX files.
        df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
        df[output_col] = [self.query(t,float(lat),float(lon))
            for t, lat, lon in zip(
                df[time_col],
                df[lat_col],
                df[lon_col])]
        return df
