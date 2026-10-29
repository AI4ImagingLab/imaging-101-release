"""Natural-spline harmonization and standard-based angular registration."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
def load_data(path):
    with np.load(path,allow_pickle=False) as f: data={k:f[k][0] for k in ['angles','counts','standards','rir']}
    shapes={'angles':(3001,),'counts':(8,3001),'standards':(3001,7),'rir':(7,)}
    for k,s in shapes.items():
        if data[k].shape!=s or data[k].dtype!=np.float64 or not np.isfinite(data[k]).all(): raise ValueError('Invalid '+k)
    return data
def natural_spline(x,y,out):
    """Natural cubic interpolation, with R's LINEAR endpoint extrapolation."""
    x=np.asarray(x); out=np.asarray(out); spline=CubicSpline(x,y,bc_type='natural',axis=0)
    bounded=np.clip(out,x[0],x[-1]); value=spline(bounded)
    extra=out-bounded
    if value.ndim>1: extra=extra[:,None]
    return value+extra*spline(bounded,1)
def align_pattern(angles,counts,standard,max_shift=.3):
    """Pearson-correlation registration; return shift and complete overlap samples."""
    centered=standard-standard.mean(); norm=np.linalg.norm(centered)
    def objective(shift):
        y=natural_spline(angles+shift,counts,angles); y=y-y.mean()
        return 1-float(centered@y/(norm*np.linalg.norm(y)))
    if max_shift==0: shift=0.
    else: shift=float(minimize_scalar(objective,bounds=(-max_shift,max_shift),method='bounded',options={'xatol':np.sqrt(np.finfo(float).eps),'maxiter':1000}).x)
    lower=max(angles[0]+shift,angles[0]); upper=min(angles[-1]+shift,angles[-1]); step=float(np.mean(np.diff(angles)))
    grid=lower+np.arange(int(np.floor((upper-lower)/step+1e-10))+1)*step
    return shift,grid,natural_spline(angles+shift,counts,grid)
