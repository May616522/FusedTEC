# IGS RT-GIM / Final-GIM 配对程序

本目录实现 2021 年 DOY 091-181 的 IGS 实时 GIM 与 IGS Final GIM 配对。

核心规则：

- 以 IONEX `EPOCH OF CURRENT MAP` 为准，不从实时文件名猜测时刻。
- 以实际存在的 20 分钟 RT 历元为主轴；RT 缺失时不补造样本。
- 完整剔除 DOY 138 和 139。
- Final 数据加载到连续时间轴，并额外读取 DOY 182 的 00:00 历元以完成 DOY 181 夜间的跨日插值。
- 按论文公式先将前后两幅 Final TEC 图以 15 deg/hour 旋转到 RT 历元，再线性插值；经度采用周期插值。
- TEC 按 IONEX `EXPONENT` 转为 TECU，经纬度统一为纬度升序、经度 -180 到 180。
- 输出残差定义为 `final_vtec - rt_vtec`。

运行（使用已有 Conda 环境）：

```powershell
D:\app\Conda\envs\ion_xgboost\python.exe F:\FusedTec\Code\GIM\match_rt_final.py --overwrite
```

运行测试：

```powershell
D:\app\Conda\envs\ion_xgboost\python.exe -m unittest F:\FusedTec\Code\GIM\test_match_rt_final.py -v
```

默认输出目录为 `F:\FusedTec\Data\GIM2021\IGS\match`，包括：

- `GIM_CNN_dataset_2021_DOY091_181.nc`
- `matching_report.csv`
- `rt_inventory.csv`
- `baseline_metrics.csv`
- `processing_summary.json`
- `qc/` 下的质量检查图

程序只负责严格匹配，不进行训练集、验证集或测试集划分。

## 初步残差 CNN

`train_residual_cnn.py` 使用已经匹配好的 NetCDF 训练轻量二维 CNN：

- 最后 3 个自然日作为严格时间外测试集；
- 更早的数据使用固定随机种子按 9:1 分为训练集和验证集；
- 输入为 RT-GIM，标签为 `Final-GIM - RT-GIM`；
- 使用 Adam、学习率 0.0005、batch size 64 和 L1 loss；
- 经度方向使用周期 padding，纬度边界使用 replicate padding；
- 根据最低验证集 L1 loss 选择最佳模型。

超算提交脚本为 `run_cnn_baseline.sh`。代码放到指定目录后可直接提交：

```bash
cd /share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN
sbatch run_cnn_baseline.sh
```

脚本默认直接调用已经检查通过的公共 PyTorch 环境：

```text
/share/apps/miniconda3/envs/tj.pytorch2.2.1/bin/python
```

该环境实际提供 PyTorch 2.5.1（CUDA 12.1），并已确认可以导入
`netCDF4`、`numpy` 和 `matplotlib`。登录节点没有 GPU 属于正常情况；提交脚本会在
GPU 计算节点上再次检查 CUDA，检查失败时会在训练开始前退出并给出错误。

批处理脚本使用固定的 `CODE_DIR` 指向代码目录，不根据 `BASH_SOURCE` 推导路径，
以避免 Slurm 从 `/var/spool/slurmd/job.../` 临时目录执行脚本时找不到训练程序。

提交脚本默认使用 `L40` 分区并申请 1 块 L40 GPU：

```text
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
```

当前 CNN 对显存需求较小，L40 的 48 GB 显存已经充足。如需改用 A800，需同时改为
`#SBATCH --partition=A800` 和 `#SBATCH --gres=gpu:a800:1`。

平台限制每块 GPU 最多申请 7 个 CPU 核，因此脚本设置
`--cpus-per-task=7`，其中 6 个用于数据加载，另 1 个留给主训练进程。

当前默认路径为：

```text
代码目录：
/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN

输入数据：
/share/home/u23114/tj23114/data/Yaoyaping_data/GIM2021/IGS/GIM_CNN_dataset_2021_DOY091_181.nc

输出目录：
/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN/results/GIM_CNN_baseline_seed42
```

所需 Python 包列在 `requirements_cnn.txt`。如果以后移动数据或希望使用其他输出目录，可以覆盖默认路径：

```bash
DATASET=/new/path/GIM_CNN_dataset.nc \
OUTPUT_DIR=/new/path/results \
sbatch run_cnn_baseline.sh
```

如需改用其他 Python 环境，可以在提交时覆盖默认路径：

```bash
PYTHON_BIN=/path/to/python sbatch run_cnn_baseline.sh
```
