from dataclasses import dataclass
import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
import tarfile
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class CosmicConfig:
    NM_F2_MIN = 2e10         # el/m^3
    NM_F2_MAX = 2e12         # el/m^3
    HM_F2_MIN = 200.0        # km
    HM_F2_MAX = 450.0        # km
    GRADIENT_MAX=-0.1e5
    MD_MAX=0.05
    MD_MIN=0
    DELTA_MAX=0.02
    
def qc_nmf2_hmf2(file,cfg):
    nm_f2_raw=file["nm_f2_raw"]
    hm_f2_raw=file["hm_f2_raw"]
    nm_f2_m3 = nm_f2_raw * 1e6
    # if not (cfg.NM_F2_MIN <= nm_f2_m3 <= cfg.NM_F2_MAX):
    #     return False, "NmF2_limit"
    if not (cfg.HM_F2_MIN <= hm_f2_raw <= cfg.HM_F2_MAX):
         return False, "hmF2_limit"
    return True,"pass"


def qc_grad(file,cfg):
    #这里面的高度是否需要排序
    alt=file["alt"]
    ne_smooth=file["ne_smooth"]
    idx_490 = np.argmin(np.abs(alt - 490.0))
    idx_420 = np.argmin(np.abs(alt - 420.0))
    if (ne_smooth[idx_490] - ne_smooth[idx_420])*1e6 / (70.0*1000) > cfg.GRADIENT_MAX:
        return False, "gradient_error"
    
    return True,"pass"

def qc_md(file,cfg):
    ne=file["ne"]
    ne_smooth=file["ne_smooth"]
    alt=file["alt"]
    mask=alt>=180
    ne=ne[mask]
    ne_smooth=ne_smooth[mask]
    n_points=len(ne_smooth)
    md = np.sum(np.abs(ne - ne_smooth) / (n_points * ne_smooth))
    if md>cfg.MD_MAX or md<cfg.MD_MIN:
        return False,"md_error"
    
    return True,"pass"

def qc_delta(file,cfg):

    ne=file["ne"]
    ne_smooth=file["ne_smooth"]
    alt=file["alt"]
    nm_f2=file["nm_f2_raw"]
    mask=alt>=300
    ne=ne[mask]
    ne_smooth=ne_smooth[mask]
    k=len(ne)
    delta=np.sqrt(
        np.sum((ne-ne_smooth)**2)/
        (k*nm_f2**2))

    if delta>cfg.DELTA_MAX:
        return False,"Delta_error"
    return True,"pass"
    

#总的质量控制函数
def quality_control(file,cfg):

    tests=[
        qc_nmf2_hmf2,
        qc_grad,
        qc_md,
        qc_delta]

    for func in tests:
        if func.__name__=="qc_nmf2_hmf2":
            ok,msg=func(file,cfg)
        if func.__name__=="qc_grad":
            ok,msg=func(file,cfg)
        if func.__name__=="qc_md":
                    ok,msg=func(file,cfg)
        if func.__name__=="qc_delta":
                    ok,msg=func(file,cfg)
        if not ok:
            return False,msg
    return True,"pass"    
     

def read_ro_profile(filepath):
    """
    读取GNSS RO电子密度剖面
    return: dict   
    """
    with xr.open_dataset(filepath) as ds:
        alt = ds["MSL_alt"].values.flatten()
        ne = ds["ELEC_dens"].values.flatten()
        #这里原本应该检查数据是否存在异常值吗？还是影响不大
        win_size = 9
        ne_smooth = uniform_filter1d(ne, size=win_size, mode="nearest")
        lat = ds["GEO_lat"].values.flatten()
        lon = ds["GEO_lon"].values.flatten()
        bot_alt = float(ds.attrs.get('botalt', 999))
        top_alt = float(ds.attrs.get('topalt', 0))
        nm_f2_raw = float(ds.attrs.get('edmax', 0)) 
        hm_f2_raw = float(ds.attrs.get('edmaxalt', 0))
    return {
        "alt":alt,
        "ne":ne,
        "ne_smooth":ne_smooth,
        "lat":lat,
        "lon":lon,
        "bot_alt": bot_alt,
        "top_alt": top_alt,
        "nm_f2_raw": nm_f2_raw,
        "hm_f2_raw": hm_f2_raw}
    
def process_file(path):
    profile=read_ro_profile(path)
    cfg=CosmicConfig()
    ok,msg=quality_control(profile,cfg)
    return {
        "file":path,
        "status":msg,
        "pass":ok
    }
    
#对一个剖面进行可视化
def plot_profile(file):

    alt=file["alt"]
    ne=file["ne"]
    ne_smooth=file["ne_smooth"]
    plt.figure(figsize=(6,8))
    plt.plot(ne,alt,label="Original Ne",linewidth=1)
    plt.plot(ne_smooth,alt,label="Smoothed Ne",linewidth=2)
    plt.xlabel("Electron density (el/cm$^3$)")
    plt.ylabel("Altitude (km)")
    plt.title("COSMIC-2 Electron Density Profile")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

def batch_test(tar_path):
    files=[]
    # 打开tar.gz
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            # 筛选nc文件
            if member.name.endswith(".0001_nc"):
                files.append(member)
        total=len(files)
        stat=defaultdict(int)
        passed=[]
        cfg=CosmicConfig()
        
        for member in tqdm(files,total=total,desc="cosmic QC",unit="profile"):
            try:
                fobj = tar.extractfile(member)
                profile = read_ro_profile(fobj)
                ok,msg = quality_control(profile,cfg)
                stat[msg]+=1
                if ok:
                    passed.append(member.name)
            except Exception as e:
                print(member.name,e)
    print("="*50)
    print("COSMIC QC RESULT")
    print("="*50)
    for k,v in stat.items():
        print(f"{k:20s}:{v}")
    print()
    print(f"Total:{total}")
    print(f"Pass:{len(passed)}")

    if total>0:
        print(f"Rate:{len(passed)/total*100:.2f}%")
    return passed

if __name__ == "__main__":
    inputpath=r"F:\FusedTec\Data\Cosmic2020\ionPrf_prov1_2020_043.tar.gz"
    profile=batch_test(inputpath)
    
