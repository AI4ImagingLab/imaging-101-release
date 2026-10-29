"""Trajectory statistics; adapted from Morgan/Saleh Tweezepy, GPL-3.0.
Greenhall EDF mathematics derives from its allantools adaptation.
"""
from pathlib import Path
import numpy as np
from scipy.fft import fft
FS=400.0
KT=4.1
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:x=z['position']
    if x.shape!=(1,13631) or x.dtype!=np.float64 or not np.isfinite(x).all():raise ValueError('Invalid position array')
    return x[0].copy()
def spectrum(trace, bins=35):
    """Two-sided periodic-Hann Welch density, positive non-Nyquist bins only."""
    x=np.asarray(trace); n=int(2*len(x)/(bins+1)); hop=n-n//2
    windows=np.array([x[i:i+n] for i in range(0,len(x)-n+1,hop)])
    window=(.5+.5*np.cos(np.linspace(-np.pi,np.pi,n+1)))[:-1]
    windows=windows-windows.mean(axis=1,keepdims=True)
    transform=fft(windows*window,axis=1)
    power=(transform.conj()*transform).real*(1/(FS*np.sum(window**2)))
    f=np.fft.fftfreq(n,1/FS);mask=f>0
    return {'x':f[mask],'y':power.mean(axis=0)[mask],'shape':np.full(mask.sum(),float(bins))}
def noise_id(trace,m):
    x=np.mean(trace[:len(trace)//m*m].reshape(-1,m),axis=1);d=0
    while True:
        r=np.corrcoef(x[1:],x[:-1])[0,1];rho=r/(1+r)
        if rho<.25 or d>=2:return int(-np.round(2*rho)-2*d)
        x=np.diff(x);d+=1
def greenhall_weight(n,m,alpha):
    """Unmodified d=2, nonoverlap EDF convention actually selected upstream.
    Here J<=3, so only original finite basic-sum branch is needed.
    """
    def sw(t,a):
        if a==2:return -abs(t)
        if a==0:return abs(t**3)
        if a==-2:return abs(t**5)
        if a in (-1,1):return 0. if t==0 else t**(3-a)*np.log(abs(t))
        raise ValueError('Unsupported power-law exponent')
    F=float(m) if 3*m<=100 else np.inf
    def sx(t):return sw(t,alpha+2) if np.isinf(F) else F**2*(2*sw(t,alpha)-sw(t-1/F,alpha)-sw(t+1/F,alpha))
    def sz(t):return 6*sx(t)-4*sx(t-1)-4*sx(t+1)+sx(t-2)+sx(t+2)
    M=1+np.floor((n-(1+2*m))/m);J=min(M,3)
    s=sz(0)**2+(1-J/M)*sz(J)**2
    for j in range(1,int(J)):s+=2*(1-j/M)*sz(j)**2
    return sz(0)**2*M/s
def allan(trace):
    """Overlapping Allan variance and original lag-one-noise EDF weighting."""
    x=np.asarray(trace);n=len(x);count=int(np.floor(np.log2(n/2)))
    ms=np.logspace(0,count-1,count,base=2,dtype=int)
    phase=np.r_[0,np.cumsum(x)/FS];ys=[];shapes=[];alpha=0
    for m in ms:
        diff=phase[2*m:]-2*phase[m:-m]+phase[:-2*m]
        ys.append(np.sum(diff*diff)/(2*len(diff)*(m/FS)**2))
        if n//m>32:alpha=noise_id(x,m)
        edf=greenhall_weight(n,m,alpha) if -3<alpha<3 else n//m-1
        shapes.append(edf/2)
    return {'x':ms/FS,'y':np.array(ys),'shape':np.array(shapes)}
