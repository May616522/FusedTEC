from dataclasses import dataclass
import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter1d
import os
from collections import defaultdict

@dataclass
class ROQCConfig:

    # cm^-3
    nmf2_min: float = 1e3
    nmf2_max: float = 1e7
    
    #km
    hmf2_min: float = 200
    hmf2_max: float = 600

    # m^-4
    gradient_threshold: float = -2e5
    mrd_threshold: float = 0.05
    delta_threshold: float = 0.03

    interp_start: float = 180
    interp_end: float = 800
    interp_step: float = 5
    
#这个函数写的不对，你时间没读
def read_ro_profile(filepath):
    """
    读取GNSS RO电子密度剖面
    return: dict
    """
    with xr.open_dataset(filepath) as ds:
        alt = ds["MSL_alt"].values.flatten()
        ne = ds["elec_Dens"].values.flatten()
        lat = ds["lat_tp"].values.flatten()
        lon = ds["lon_tp"].values.flatten()
        win_size = 9
        ne_smooth = uniform_filter1d(ne, size=win_size, mode="nearest")
    return {
        "alt":alt,
        "ne":ne,
        "lat":lat,
        "lon":lon,
        "ne_smooth":ne_smooth}

#单个电子密度剖面，Ne<100 cm−3 或 Ne>2×10^7 cm−3 删除  
#对这个函数存疑，是否要这么处理，还是只有到tec计算的时候才这样 
def preprocess_profile(profile):
    alt = profile["alt"]
    ne = profile["ne"]
    mask = (np.isfinite(alt)&np.isfinite(ne))
    alt = alt[mask]
    ne = ne[mask]
    mask = ((ne>=100)&(ne<=2e7))
    alt = alt[mask]
    ne = ne[mask]
    # 高度排序
    idx=np.argsort(alt)
    alt=alt[idx]
    ne=ne[idx]
    return alt,ne

def get_nmf2_hmf2(alt,ne):
    idx=np.argmax(ne)
    nmf2=ne[idx]
    hmf2=alt[idx]
    return nmf2,hmf2

def qc_nmf2_hmf2(alt,ne,cfg):
    nmf2,hmf2=get_nmf2_hmf2(alt,ne)
    if not (cfg.nmf2_min<=nmf2<=cfg.nmf2_max):
        return False,"NmF2"
    if not(cfg.hmf2_min<=hmf2<=cfg.hmf2_max):
        return False,"hmF2"
    return True,"pass"


#对这个五点平均存在质疑，但是应该影响比较小，对于整个文件来说
def smooth_ne(ne):
    return uniform_filter1d(ne,size=5)

#垂直梯度检查
def gradient_test(alt,ne,hmf2,cfg):
    if hmf2 > 420:
        return True,"gradient_skip"
    ne_s = smooth_ne(ne)
    ne420 = np.interp(420,alt,ne_s)
    ne490 = np.interp(490,alt,ne_s)
    # cm^-3 -> m^-3
    ne420_m3 = ne420 * 1e6
    ne490_m3 = ne490 * 1e6

    # km -> m
    grad = (ne490_m3 - ne420_m3) / (70*1000)
    if grad < cfg.gradient_threshold:
        return False,"gradient"
    return True,"pass"

def calculate_mrd(ne,alt,ne_smooth):
    
    mask=alt>=180
    ne=ne[mask]
    ne_smooth=ne_smooth[mask]
    n_points=len(ne_smooth)
    mrd=np.mean(np.abs((ne-ne_smooth)/(ne_smooth*n_points)))
    return mrd

def mrd_test(ne,alt,ne_smooth,cfg):
    mrd=calculate_mrd(ne,alt,ne_smooth)
    if mrd>cfg.mrd_threshold:
        return False,"MRD"
    return True,"pass"

def delta_test(alt,ne,cfg):

    nmf2,_=get_nmf2_hmf2(alt,ne)
    mask=alt>=300
    if np.sum(mask)<10:
        return False,"Delta_sample"
    ne300=ne[mask]
    smooth=uniform_filter1d(ne,9)
    smooth300=smooth[mask]
    N=len(ne300)
    delta=np.sqrt(np.sum((ne300-smooth300)**2)/(N*nmf2**2))
    if delta>cfg.delta_threshold:
        return False,"Delta"
    return True,"pass"



def quality_control(alt,ne,ne_smooth,cfg):
    nmf2,hmf2=get_nmf2_hmf2(alt,ne)
    ok,msg=qc_nmf2_hmf2(alt,ne,cfg)
    if not ok:
        return False,msg
    ok,msg=gradient_test(alt,ne,hmf2,cfg)
    if not ok:
        return False,msg
    ok,msg=mrd_test(ne,alt,ne_smooth,cfg)
    if not ok:
        return False,msg
    ok,msg=delta_test(alt,ne,cfg)
    if not ok:
        return False,msg

    return True,"pass"

def process_file(path):
    profile=read_ro_profile(path)
    #alt,ne=preprocess_profile(profile )
    ne=profile["ne"]
    ne_smooth=profile["ne_smooth"]
    alt=profile["alt"]
    cfg=ROQCConfig()
    ok,msg=quality_control(alt,ne,ne_smooth,cfg)
    return {
        "file":path,
        "status":msg,
        "pass":ok
    }

from scipy.optimize import curve_fit


#α-Chapman函数
def alpha_chapman(h,nmf2,hmf2,Hm,a1,a2):

    H=np.where( h>hmf2,a1*(h-hmf2)+Hm,a2*(h-hmf2)+Hm)
    z=(h-hmf2)/H
    ne=nmf2*np.exp(0.5*(1-z-np.exp(-z)))
    return ne

#拟合α-Chapman函数
def fit_chapman(alt,ne):

    mask=alt>=180
    h=alt[mask]
    y=ne[mask]
    nmf2,hmf2=get_nmf2_hmf2(h,y)
    p0=[nmf2, hmf2,50,0.001,-0.001]
    bounds=([1e3,100,1,-1,-1],[1e7,800,500,1,1])

    popt,_=curve_fit(alpha_chapman,h,y,p0=p0,bounds=bounds,maxfev=10000)
    return popt


def batch_test(folder):

    files=[]
    for f in os.listdir(folder):
        if f.endswith(".NC"):
            files.append(
                os.path.join(folder,f))
    total=len(files)
    stat=defaultdict(int)
    passed=[]
    cfg=ROQCConfig()
    for f in files:
        try:
            profile=read_ro_profile(f)
            ne=profile["ne"]
            ne_smooth=profile["ne_smooth"]
            alt=profile["alt"]
            ok,msg=quality_control(alt,ne,ne_smooth,cfg)
            stat[msg]+=1
            if ok:

                passed.append(f)


        except Exception as e:

             print(f,e)



    print("="*50)

    print("FY3E QC RESULT")

    print("="*50)


    for k,v in stat.items():

        print(
            f"{k:20s}:{v}"
        )


    print()

    print(
        f"Total:{total}"
    )

    print(
        f"Pass:{len(passed)}"
    )

    print(
        f"Rate:{len(passed)/total*100:.2f}%"
    )


    return passed


def debug_one(path):

    profile=read_ro_profile(path)


    alt=profile["alt"]
    ne=profile["ne"]


    print("----------------")

    print("height:")
    print(
        alt.min(),
        alt.max()
    )


    print("Ne:")
    print(
        ne.min(),
        ne.max()
    )


    alt,ne=preprocess_profile(
        profile
    )


    nmf2,hmf2=get_nmf2_hmf2(
        alt,
        ne
    )


    print(
        "NmF2:",
        nmf2
    )

    print(
        "hmF2:",
        hmf2
    )
    
if __name__ == "__main__":
    inputpath=r"F:\FusedTec\Data\FY2020\M1\A2026080802270264990001"
    profile=batch_test(inputpath)