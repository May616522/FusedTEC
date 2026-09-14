#对于提供的.dat文件，按照列分割，并存储到csv文件中

import pandas as pd
import numpy as np
# 1. 定义 55 个列名（对应数据格式说明表）
columns = [
    "Year",                       # 1. 年份
    "Decimal_Day",                 # 2. 积日 (Jan 1 = 1)
    "Hour",                        # 3. 小时 (0-23)
    "Bartels_Rotation_Num",        # 4. Bartels 旋转周号
    "ID_IMF_Spacecraft",           # 5. IMF 卫星 ID
    "ID_SW_Plasma_Spacecraft",     # 6. 太阳风等离子体卫星 ID
    "Num_Points_IMF_Avg",          # 7. IMF 平均点数
    "Num_Points_Plasma_Avg",       # 8. 等离子体平均点数
    "Field_Mag_Avg_B",             # 9. 磁场强度均值 |B| (nT)
    "Mag_Avg_Field_Vector",        # 10. 平均磁场矢量大小 (nT)
    "Lat_Angle_Avg_Field_Vector",  # 11. 平均磁场矢量纬角 (deg)
    "Long_Angle_Avg_Field_Vector", # 12. 平均磁场矢量经角 (deg)
    "Bx_GSE_GSM",                  # 13. Bx GSE/GSM (nT)
    "By_GSE",                      # 14. By GSE (nT)
    "Bz_GSE",                      # 15. Bz GSE (nT)
    "By_GSM",                      # 16. By GSM (nT)
    "Bz_GSM",                      # 17. Bz GSM (nT)
    "sigma_B_mag",                 # 18. |B| 标准差 (nT)
    "sigma_B_vec",                 # 19. 磁场矢量标准差 (nT)
    "sigma_Bx",                    # 20. Bx 标准差 (nT)
    "sigma_By",                    # 21. By 标准差 (nT)
    "sigma_Bz",                    # 22. Bz 标准差 (nT)
    "Proton_Temperature",          # 23. 质子温度 (K)
    "Proton_Density",              # 24. 质子密度 (N/cm^3)
    "Plasma_Speed",                # 25. 等离子体流速 (km/s)
    "Plasma_Flow_Long_Angle",      # 26. 流速经角 (deg)
    "Plasma_Flow_Lat_Angle",       # 27. 流速纬角 (deg)
    "Na_Np_Ratio",                 # 28. 氦/质子密度比 Na/Np
    "Flow_Pressure",               # 29. 动压 Flow Pressure (nPa)
    "sigma_T",                     # 30. 温度标准差 (K)
    "sigma_N",                     # 31. 密度标准差 (N/cm^3)
    "sigma_V",                     # 32. 速度标准差 (km/s)
    "sigma_phi_V",                 # 33. 速度经角标准差 (deg)
    "sigma_theta_V",               # 34. 速度纬角标准差 (deg)
    "sigma_Na_Np",                 # 35. Na/Np 标准差
    "Electric_Field",              # 36. 电场强度 (mV/m)
    "Plasma_Beta",                 # 37. 等离子体 β 值
    "Alfven_Mach_Number",          # 38. 阿尔芬马赫数
    "Kp_Index",                    # 39. Kp 地磁指数 (*10)
    "Sunspot_Number_R",            # 40. 太阳黑子数 R
    "Dst_Index",                   # 41. Dst 指数 (nT)
    "AE_Index",                    # 42. AE 指数 (nT)
    "Proton_Flux_gt_1MeV",         # 43. 质子通量 >1 MeV
    "Proton_Flux_gt_2MeV",         # 44. 质子通量 >2 MeV
    "Proton_Flux_gt_4MeV",         # 45. 质子通量 >4 MeV
    "Proton_Flux_gt_10MeV",        # 46. 质子通量 >10 MeV
    "Proton_Flux_gt_30MeV",        # 47. 质子通量 >30 MeV
    "Proton_Flux_gt_60MeV",        # 48. 质子通量 >60 MeV
    "Flag",                        # 49. 标识 Flag (-1~6)
    "ap_Index",                    # 50. ap 地磁指数 (nT)
    "f10.7_Index",                 # 51. F10.7 太阳辐射通量指数
    "PC_N_Index",                  # 52. PC(N) 北极极冠指数
    "AL_Index",                    # 53. AL 指数 (nT)
    "AU_Index",                    # 54. AU 指数 (nT)
    "Magnetosonic_Mach_Number"     # 55. 磁声波马赫数
]

# 2. 读取 dat 文件并导出为 CSV
input_dat_path = "F:\FusedTec\Data\Other\omni2_2021.dat.txt"  # 请替换为你的 .dat 文件路径
output_csv_path =  "F:\FusedTec\Data\Other\omni2_2021.dat.csv"  # 导出的 CSV 文件名

# sep=r'\s+' 表示按任意数量的空白字符分隔
df = pd.read_csv(input_dat_path, sep=r'\s+', names=columns, header=None)

df['f10.7_Index'] = df['f10.7_Index'].replace(999.9, np.nan)
df['Dst_Index'] = df['Dst_Index'].replace(99999, np.nan)
df['Kp_Index'] = df['Kp_Index'].replace(99, np.nan)
df['Kp_Index'] = df['Kp_Index'] / 10.0

df=df[["Year","Decimal_Day","Hour","f10.7_Index","Dst_Index","Kp_Index"]]
# 3. 保存为 CSV 文件
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"解析完成！已成功转换并保存至：{output_csv_path}")