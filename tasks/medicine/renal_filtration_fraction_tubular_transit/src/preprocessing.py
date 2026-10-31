"""Strict public arrays, real measured curves and independent calibration."""
from pathlib import Path
import numpy as np
NCASES=228
MAX_TIME=125
HCT=.45

def load_data(path):
    with np.load(Path(path),allow_pickle=False) as a: data={k:a[k].copy() for k in a.files}
    for k in ['measurements','arterial_signal','time']:
        if data[k].shape!=(1,NCASES,MAX_TIME) or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all():raise ValueError('Invalid '+k)
    for i,L in enumerate(data['lengths'][0]):
        if not 3<=L<=MAX_TIME or not np.all(np.diff(data['time'][0,i,:L])>0):raise ValueError('Invalid sampled time')
        if not 1<=data['baseline_count'][0,i]<L:raise ValueError('Invalid baseline count')
    return data

def case_data(data,index):
    """Return times, measured renal signal, aortic signal, calibration dict."""
    L=int(data['lengths'][0,index])
    cal={k:float(data[k][0,index]) for k in ['tr','flip_angle','tissue_t1','blood_t1','relaxivity','volume','field_strength']}
    cal['n0']=int(data['baseline_count'][0,index]);cal['H']=HCT
    return data['time'][0,index,:L],data['measurements'][0,index,:L],data['arterial_signal'][0,index,:L],cal
