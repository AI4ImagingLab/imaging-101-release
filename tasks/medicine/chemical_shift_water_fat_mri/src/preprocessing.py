"""Bound acquisition loading and selected author-model parameters."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
SHAPE=(1,3,101,101,4)
TE=np.array([.00287,.00607,.00927],dtype=np.float64)
B0=1.494
GYRO=42.576
def load_observations(path: str|Path='data/raw_data.npz') -> dict:
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'echoes'}:raise ValueError('raw observations require only echoes')
        a=archive['echoes']
    if a.shape!=SHAPE or a.dtype!=np.complex64 or not np.isfinite(a).all():raise ValueError('invalid finite complex64 echo data')
    return {'echoes':a}
def parameters(data: dict,mu: float=.1,multiscale: bool=True,icm_iterations: int=10,algorithm: str='QPBO') -> tuple:
    d=SimpleNamespace(data=data['echoes'][0].copy(),N=3,dt=float(np.diff(TE).mean()),t1=float(TE[0]),B0=B0,voxelsize=(1.5,1.5,5.),cyclic=(False,False,False),offresCenter=0.)
    a=SimpleNamespace(nR2=145,R2step=1.,nB0=100,iR2cand=np.array([40]),mu=mu,offresPenalty=0.,neighbourhoodRadius=0.,multiScale=multiscale,nICMiter=icm_iterations,maxICMupdate=10,graphcutLimit=0.,algorithm=algorithm)
    cs=np.array([1.3+3.748-.01085*37,5.3,4.31,2.76,2.1,1.3,.9],dtype=np.float32)
    alpha=np.zeros((2,7),dtype=np.float32);alpha[0,0]=1;alpha[1,1:]=[.048,.039,.004,.128,.693,.087]
    m=SimpleNamespace(CS=cs,alpha=alpha,realEstimates=False,M=2)
    return d,a,m
