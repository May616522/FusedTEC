import numpy as np

#统计两个列之间的一些指标
def calculate_vtec_stats(df,compare_col,gim_col="gim_vtec"):
    data = df[[gim_col,compare_col]].copy()
    data = data.replace([np.inf, -np.inf],np.nan)
    data = data.dropna()

    if len(data) == 0:
        raise ValueError(
            "没有有效数据")
    gim = data[gim_col].to_numpy()
    obs = data[compare_col].to_numpy()
    residual = (gim - obs)
    stats = {"correlation":np.corrcoef(gim,obs)[0, 1],
            "bias":np.mean(residual),
            "std":np.std(residual,ddof=1),
            "rmse":np.sqrt(np.mean(residual ** 2)),
            "mae":np.mean(np.abs(residual)),
            "n":len(data)}
    return stats

#利用两个列之间的关系进行绘图
