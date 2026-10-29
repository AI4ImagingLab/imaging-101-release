"""GPL-3.0-or-later: explicit extraction of Atomap preprocessing mathematics.

Atomap contributors; numerical decomposition follows the BSD-licensed
scikit-learn randomized range algorithm, expressed using NumPy/SciPy only.
"""
import numpy as np
from scipy import linalg,ndimage
PIXEL_NM=.012443924564929288
SHAPE=(1,811,913)
def load_data(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for k in ['adf','abf']:
            a=z[k]
            if a.shape!=SHAPE or a.dtype!=np.dtype('float64') or not np.isfinite(a).all() or np.any(a<=0):
                raise ValueError(f'Invalid positive float64 {k} observations')
            out[k]=a[0].copy()
    return out
def filter_adf(image,seed=20261022):
    a=np.asarray(image,dtype=np.float64)
    a=a-ndimage.gaussian_filter(a,30,mode='nearest');a-=a.min()
    m=a.T
    rng=np.random.RandomState(seed)
    q=rng.normal(size=(m.shape[1],32))
    for _ in range(5):
        q,_=linalg.lu(m@q,permute_l=True,check_finite=False)
        q,_=linalg.lu(m.T@q,permute_l=True,check_finite=False)
    q,_=linalg.qr(m@q,mode='economic',check_finite=False)
    _,_,v=linalg.svd(q.T@m,full_matrices=False,lapack_driver='gesdd')
    v=v[:22]
    b=((m@v.T)@v).T
    if b.min()<0:b-=b.min()
    return b/b.max()
def find_peaks(image,separation=12):
    # Maximum-filter candidate detection, descending intensity and infinity-norm
    # exclusion implement the selected reference peak convention directly.
    a=np.asarray(image);s=int(separation)
    mask=(a==ndimage.maximum_filter(a,size=2*s+1,mode='nearest'))&(a>.02*a.max())
    mask[:s]=False;mask[-s:]=False;mask[:,:s]=False;mask[:,-s:]=False
    p=np.argwhere(mask);p=p[np.argsort(-a[tuple(p.T)],kind='stable')]
    keep=[]
    for xy in p:
        if not keep or np.min(np.max(abs(np.asarray(keep)-xy),axis=1))>=s:keep.append(xy)
    return np.asarray(keep)[:,::-1]
