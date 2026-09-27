from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

prediction_file = Path(
    r"F:\FusedTec\Results\xg_jason_results"
    r"\xg_Jason_ion_results8\all_split_predictions.csv"
)

columns = [
    "split",
    "true_residual",
    "predicted_residual",
    "true_jason_tec",       # 这里实际对应 TEC_smooth
    "gim_vtec_baseline",    # 这里实际对应 gim_vtec
]

data = pd.read_csv(prediction_file, usecols=columns)

# 验证 residual 的定义和符号方向
formula_error = (
    data["true_residual"]
    - (data["gim_vtec_baseline"] - data["true_jason_tec"])
)
print("Residual 公式最大误差:", formula_error.abs().max())

for split in ["train", "validation", "test"]:
    frame = data.loc[data["split"] == split]

    true_residual = frame["true_residual"].to_numpy()
    predicted_residual = frame["predicted_residual"].to_numpy()
    tec_smooth = frame["true_jason_tec"].to_numpy()
    gim_vtec = frame["gim_vtec_baseline"].to_numpy()

    corrected_gim = tec_smooth + predicted_residual

    # 两种评价的误差理论上应完全相同
    residual_error = predicted_residual - true_residual
    corrected_error = corrected_gim - gim_vtec

    print(f"\n{split}:")
    print("样本数:", len(frame))
    print(
        "误差一致性最大偏差:",
        np.max(np.abs(residual_error - corrected_error)),
    )
    print(
        "Residual R²:",
        r2_score(true_residual, predicted_residual),
    )
    print(
        "TEC_smooth 基线 R²:",
        r2_score(gim_vtec, tec_smooth),
    )
    print(
        "校正后 GIM R²:",
        r2_score(gim_vtec, corrected_gim),
    )
    print(
        "校正后 RMSE:",
        mean_squared_error(gim_vtec, corrected_gim) ** 0.5,
    )
    print(
        "校正后 MAE:",
        mean_absolute_error(gim_vtec, corrected_gim),
    )