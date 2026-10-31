"""Visible PTA observations and explicit author processing convention."""
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
COORDS=np.linspace(-85.,85.,100)

def load_observations(path):
    with np.load(Path(path),allow_pickle=False) as z:
        keys={'raw_coordinates','raw_signal','sample_mask','profile_mean','profile_sigma','transfer_matrices'}
        if set(z.files)!=keys:raise ValueError('Unexpected observation keys')
        arrays={k:z[k].copy() for k in keys}
    c,y,mask=[arrays[k] for k in ['raw_coordinates','raw_signal','sample_mask']]
    if c.ndim!=4 or c.shape[:3]!=(1,8,3) or y.shape!=c.shape or mask.shape!=c.shape or mask.dtype!=np.bool_:raise ValueError('Invalid native profile shape/mask')
    for k in keys-{'sample_mask'}:
        if arrays[k].dtype!=np.float64 or not np.isfinite(arrays[k]).all():raise ValueError('Native observations must preserve finite float64 precision')
    if arrays['profile_mean'].shape!=(1,8,3) or arrays['profile_sigma'].shape!=(1,8,3) or arrays['transfer_matrices'].shape!=(1,8,4,4):raise ValueError('Invalid moment or transport shapes')
    if (arrays['profile_sigma']<=0).any():raise ValueError('Nonpositive observed RMS width')
    output={'transfer_matrices':arrays['transfer_matrices'][0]}
    for i in range(8):
        for j,dim in enumerate(['x','y','u']):
            valid=mask[0,i,j]
            if valid.sum()<3 or not np.array_equal(valid,np.arange(len(valid))<valid.sum()):raise ValueError('Expected contiguous native acquisition samples')
            key=f'm{i//4}_w{i%4}_{dim}'
            output[key+'_coords']=c[0,i,j,valid]
            output[key+'_counts']=y[0,i,j,valid]
            output[key+'_mean']=arrays['profile_mean'][0,i,j]
            output[key+'_sigma']=arrays['profile_sigma'][0,i,j]
    return output

def preprocess(raw):
    values=[];variances=[]
    for m in range(2):
        for w in range(4):
            for direction in ['x','y','u']:
                key=f'm{m}_w{w}_{direction}'
                y=raw[key+'_counts'].copy();c=raw[key+'_coords'].copy()
                c-=float(raw[key+'_mean'])
                if direction=='y':c=-c[::-1];y=y[::-1]
                threshold=.01*y.max()
                if (y<0).any():y-=y.min()
                y[y<threshold]=0
                bad=np.r_[False,np.abs(np.diff(y)/y.max())>.25]
                c=c[~bad];y=y[~bad]
                y=interp1d(c,y,kind='linear',bounds_error=False,fill_value=0.)(COORDS)
                y=gaussian_filter(y,1.)
                y-=y.min();y/=y.sum()*(COORDS[1]-COORDS[0])
                values.append(y)
                variances.append(float(raw[key+'_sigma'])**2)
    return np.asarray(values).reshape(8,3,100),np.asarray(variances).reshape(8,3)

