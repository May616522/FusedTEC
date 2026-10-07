# Sentinel-6 多文件 TEC/GIM 批处理

`sentinel6_tec_batch.py` 默认处理 2021 年全年目录中的全部 Sentinel-6 `.nc` 文件，执行基础海洋 QC，并输出：

- `tec_below_raw_tecu`：由未滤波双频改正换算的轨道以下 TEC；
- `tec_below_filtered_tecu`：由数值重跟踪器滤波改正 `iono_cor_alt_filtered_nr` 换算的轨道以下 TEC（正式分析量）；
- `tec_below_filtered_mle4_tecu`：由普通 MLE4 滤波改正 `iono_cor_alt_filtered` 换算的诊断量；
- `tec_topside_corrected_tecu`：`tec_below_filtered_tecu / 0.881`；
- `tec_official_tecu`：Sentinel-6 直接提供的全柱 TEC；
- `tec_product_gim_tecu`：由产品内置 `iono_cor_gim` 换算的 GIM TEC；
- `tec_jpl_final_gim_tecu`：从外部 JPL FINAL IONEX 时空插值得到的 GIM VTEC；
- `tec_igs_gim_tecu`：从外部 IGS IONEX 时空插值得到的 GIM VTEC。

默认输入、GIM 和结果目录分别为：

```text
F:\FusedTec\Data\Sentinel2021\LR_G01_reduced
F:\FusedTec\Data\GIM2021\IGS\Final
F:\FusedTec\Results\Sentinel2021_FullYear_TEC_IGS_Final
```

在项目使用的 Conda 环境中运行：

```powershell
conda run -n ion_xgboost python Code\sentinel\sentinel6_tec_batch.py
```

快速试跑前 5 个文件：

```powershell
conda run -n ion_xgboost python Code\sentinel\sentinel6_tec_batch.py `
  --max-files 5 `
  --end-date 2021-01-01 `
  --output-dir Results\Sentinel2021_FullYear_TEC_IGS_Final_smoke
```

主要结果：

```text
daily/                 使用 --write-daily-observations 时写出每日观测级 CSV
file_audit.csv         每个 nc 文件的处理与异常记录
daily_summary.csv      每日各 TEC 的均值和样本数
daily_metrics.csv      每日 R、R²、Bias、RMSE、MAE、STD
annual_metrics.csv     全年总体指标
figures/               IGS回归图、逐日 Bias/RMSE 图及逐日指标图
```

指标表中参考量和估计量分别写在 `reference`、`estimate` 字段。Bias 始终定义为：

```text
Bias = estimate - reference
```

`R²` 使用预测意义上的决定系数 `1 - SSE/SST`，并另外提供 Pearson `R`，两者不会混用。

观测级 CSV 同时保留两个方向的残差，列名直接写明符号：

- `residual_official_minus_igs_gim_tecu`：Sentinel-6 official TEC − IGS GIM；
- `residual_igs_gim_minus_official_tecu`：IGS GIM − Sentinel-6 official TEC；
- JPL FINAL 和产品内置 GIM 也提供明确命名的残差列。

正式指标和散点图只比较以下两组关系：

1. 外部 IGS GIM 与 `iono_cor_alt_filtered_nr` 直接换算的轨道以下 TEC；
2. 外部 IGS GIM 与 `total_electron_content`。

`daily_bias_rmse.png` 单独绘制两组比较的逐日 Bias 和 RMSE。Bias 定义为
`IGS Final GIM - Sentinel-6 TEC`。`daily_r2_rmse_bias.png` 另外保留逐日
R²、RMSE 和 Bias 的综合图；`daily_bias_rmse_distribution.png` 给出全年
逐日 Bias、RMSE 的频率分布。为避免全年观测级 CSV 占用数 GB 空间，默认
不写 `daily/` 文件；确实需要时增加 `--write-daily-observations`。

JPL FINAL 现在是可选输入。需要同时生成 JPL 结果时，显式传入
`--jpl-gim-dir F:\FusedTec\Data\GIM2021\JPL\FINAL`。

散点图横轴固定为 `reference`，纵轴固定为 `estimate`，因此图中 Bias 与指标表一致，均为纵轴减横轴。

IONEX 插值会读取实际经度网格起点，自动兼容 `-180…180°` 和 `0…360°`。对于未重复末端经线的全球网格（如 `0…355°`），程序会补入首列作为 `360°` 接缝，并保证输入经度加减任意个 `360°` 后得到相同结果。

`TEC_FRACTION_BELOW_ORBIT = 0.881` 是轨道以下 TEC 占全柱 TEC 的比例，不是指数。F08/G01 官方 `total_electron_content` 使用 `iono_cor_alt_filtered_nr / 0.881`。普通 MLE4 的 `iono_cor_alt_filtered` 与官方 TEC 并非完全同源，不能通过把 0.881 调成 0.9 来替代字段选择。
