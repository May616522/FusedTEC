import numpy as np
import re
from datetime import datetime
import unlzw3
from pathlib import Path

"""
这个文件只实现GIM产品的VTEC读取
"""

#从字符串 line 中，提取所有整数、浮点数（带正负号），返回字符串形式的数字列表
def _find_numbers(line):
    return re.findall(r'[+-]?\d+\.\d+|[+-]?\d+', line)

#从ionex读取TEC数据
def parse_ionex_grid(file_path):
    """
        TEC 数据，shape = (time, lat, lon)
        单位：TECU
    """

    with open(file_path, "rb") as f:
        data = f.read()
    text = unlzw3.unlzw(data).decode("utf-8",errors="ignore")
    lines = text.splitlines()

    lat1 = None
    lat2 = None
    dlat = None
    lon1 = None
    lon2 = None
    dlon = None
    exponent = 0
    header_end = None

    for i, line in enumerate(lines):
        if "EPOCH OF FIRST MAP" in line:
            nums = _find_numbers(line)
            if len(nums) >= 6:base_time = datetime(*map(int, nums[:6]))

        elif "INTERVAL" in line:
            nums = _find_numbers(line)
            if nums:
                interval = int(nums[0])
   
        elif "LAT1 / LAT2 / DLAT" in line:
            nums = _find_numbers(line)
            if len(nums) >= 3:
                lat1 = float(nums[0])
                lat2 = float(nums[1])
                dlat = float(nums[2])

        elif "LON1 / LON2 / DLON" in line:
            nums = _find_numbers(line)
            if len(nums) >= 3:
                lon1 = float(nums[0])
                lon2 = float(nums[1])
                dlon = float(nums[2])

        elif "EXPONENT" in line:
            nums = _find_numbers(line)
            if nums:
                exponent = int(nums[0])
                
        elif "END OF HEADER" in line:
            header_end = i
            break


    # 所以不能简单用 min/max，否则会导致TEC矩阵南北翻转。


    nlat = int(round((lat2 - lat1) / dlat)) + 1
    nlon = int(round((lon2 - lon1) / dlon)) + 1
    lats = lat1 + np.arange(nlat) * dlat
    lons = lon1 + np.arange(nlon) * dlon
    scale = 10.0 ** exponent

    maps = []
    times = []
    i = header_end + 1
    while i < len(lines):
        if "START OF TEC MAP" not in lines[i]:
            i += 1
            continue
        i += 1
        while (i < len(lines)and "EPOCH OF CURRENT MAP" not in lines[i]):
            if "END OF TEC MAP" in lines[i]:
                break
            i += 1
        if i >= len(lines):
            break
        if "EPOCH OF CURRENT MAP" not in lines[i]:
            i += 1
            continue
        nums = _find_numbers(lines[i])
        if len(nums) < 6:raise ValueError(f"无法解析TEC MAP时间：{lines[i]}")
        epoch = datetime(*map(int, nums[:6]))
        i += 1
        grid = []
        while (i < len(lines)and "END OF TEC MAP" not in lines[i]):
            if "LAT/LON1/LON2/DLON/H" in lines[i]:
                header_nums = _find_numbers(lines[i])
                if len(header_nums) < 4:raise ValueError( f"无法解析纬度行：{lines[i]}" )
                current_lat = float(header_nums[0])
                current_lon1 = float(header_nums[1])
                current_lon2 = float(header_nums[2])
                current_dlon = float(header_nums[3])
                row_nlon = int(round((current_lon2 - current_lon1)/ current_dlon)) + 1
                row = []
                i += 1
                while (
                    i < len(lines)
                    and "LAT/LON1/LON2/DLON/H"
                    not in lines[i]
                    and "END OF TEC MAP"
                    not in lines[i]
                ):
                    values = _find_numbers(lines[i])
                    for value in values:
                        tec = float(value) * scale
                        row.append(tec)
                    i += 1

                if len(row) != row_nlon:
                    raise ValueError(
                        f"{epoch} "
                        f"纬度 {current_lat}° TEC数量错误："
                        f"应该有 {row_nlon} 个，"
                        f"实际读取 {len(row)} 个"
                    )

                grid.append(row)

            else:
                i += 1
        grid = np.asarray(grid,dtype=float)

        # 检查纬度数量
        if grid.shape != (nlat, nlon):
            raise ValueError(
                f"{epoch} TEC MAP维度错误："
                f"期望 {(nlat, nlon)}，"
                f"实际 {grid.shape}"
            )
        maps.append(grid)
        times.append(epoch)
        if ( i < len(lines) and "END OF TEC MAP" in lines[i]):
            i += 1

    if len(maps) == 0:raise ValueError("文件中没有读取到任何 TEC MAP")
    cube = np.stack(maps,axis=0)
    return times, lats, lons, cube

if __name__== "__main__":
    path = Path(r"F:\FusedTec\Data\GIM2020\COD\001\codg0010.20i.Z")
    parse_ionex_grid(path)
    