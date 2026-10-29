"""Public measured spectra and independent calibration proteins."""
import numpy as np
def load_data(path='data/raw_data.npz'):
    shapes={'observations':(1,4,267),'library_spectra':(1,24,267),'library_fractions':(1,24,4)}
    with np.load(path,allow_pickle=False) as f:
        result={k:f[k].copy() for k in shapes}
    for k,a in result.items():
        if a.shape!=shapes[k] or a.dtype!=np.float32 or not np.isfinite(a).all(): raise ValueError('Malformed spectral input '+k)
    return result
def prepare(data,ir_scale=15.,mode='joint'):
    A=data['library_spectra'][0].T.copy();q=data['observations'][0].T.copy();F=data['library_fractions'][0].T.copy()
    if mode=='ir':return A[66:],F,q[66:]
    if mode!='joint':raise ValueError('mode must be joint or ir')
    if ir_scale==0:return A[:66],F,q[:66]
    select=np.r_[np.arange(66),np.arange(66,187,2)]
    A=A[select];q=q[select];A[66:]*=ir_scale;q[66:]*=ir_scale
    return A,F,q
