"""Read original observations and implement the published finite-window resampling."""
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .physics_model import TARGET_E
def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as z:
        expected={'intensity','variance','measured_mask','q_axis','energy_axis'}
        if set(z.files)!=expected:raise ValueError('Unexpected observation keys')
        arrays={k:z[k].copy() for k in expected}
    if arrays['intensity'].shape!=(1,300,149) or arrays['variance'].shape!=(1,300,149):raise ValueError('Expected original300x149 detector grid')
    if arrays['measured_mask'].shape!=(1,300,149) or arrays['measured_mask'].dtype!=np.bool_:raise ValueError('Invalid acquisition mask')
    for k in ['intensity','variance','q_axis','energy_axis']:
        if arrays[k].dtype!=np.float64:raise ValueError('Preserve original float64 precision')
    valid=arrays['measured_mask']
    if not np.array_equal(valid,np.isfinite(arrays['intensity'])):raise ValueError('Acquisition mask mismatch')
    if not np.isfinite(arrays['variance'][valid]).all() or (arrays['variance'][valid]<0).any():raise ValueError('Invalid measured variance')
    if arrays['q_axis'].shape!=(1,300) or arrays['energy_axis'].shape!=(1,149):raise ValueError('Invalid original axes')
    if not np.isfinite(arrays['q_axis']).all() or not np.isfinite(arrays['energy_axis']).all():raise ValueError('Invalid energy or momentum coordinate')
    return tuple(arrays[k][0] for k in ['intensity','variance','q_axis','energy_axis'])

def preprocess(intensity,variance,q,energy):
    mask=~np.isfinite(intensity)
    a=np.where(mask,0.,intensity);v=np.where(mask,0.,variance)
    observed=RectBivariateSpline(energy,q,a.T,kx=1,ky=1)(TARGET_E,q).T
    errors=RectBivariateSpline(energy,q,v.T,kx=1,ky=1)(TARGET_E,q).T
    for i,row in enumerate(~mask):
        idx=np.flatnonzero(row)
        left,right=(energy[idx[0]],energy[idx[-1]]) if len(idx) else (energy[0],energy[0])
        if left>TARGET_E[0]:
            j=int(np.rint(min(left,TARGET_E[-1])-TARGET_E[0]))
            observed[i,:j+1]=np.nan;errors[i,:j+1]=np.nan
        if right<TARGET_E[-1]:
            j=int(np.rint(max(right,TARGET_E[0])-TARGET_E[0]))
            observed[i,j:]=np.nan;errors[i,j:]=np.nan
    return observed,errors

