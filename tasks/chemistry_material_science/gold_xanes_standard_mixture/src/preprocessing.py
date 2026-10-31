"""Measured standards retain independent energy axes; interpolate linearly."""
from pathlib import Path
import numpy as np
NAMES=('unknown','s1','s2','s3','s4','s5','s6')
COUNTS=(327,327,327,308,327,327,324)
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as z:
        out={k:z[k].copy() for k in NAMES}
    for k,n in zip(NAMES,COUNTS):
        a=out[k]
        if a.shape!=(1,n,2) or a.dtype.kind!='f' or not np.isfinite(a).all() or not np.all(np.diff(a[0,:,0])>0):
            raise ValueError('Invalid measured spectrum '+k)
    return out
def prepare(raw):
    a=raw['unknown'][0];e=a[:,0]
    lo=np.searchsorted(e,11870.,side='right')-1
    hi=np.searchsorted(e,12030.,side='right')-1
    energy=e[lo:hi+1].copy(); observation=a[lo:hi+1,1].copy()
    standards=[]
    for k in NAMES[1:]:
        x,y=raw[k][0].T
        if energy[0]<x[0] or energy[-1]>x[-1]:raise ValueError('Extrapolation outside measured standard')
        standards.append(np.interp(energy,x,y))
    return energy,observation,np.array(standards)
