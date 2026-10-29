"""Observation handling and independent fluorescent-lake calibration.

Adapted mathematical stages from Polaris, MIT license in task LICENSE.
No target-derived calibration or intensity clipping is performed.
"""
from pathlib import Path
import numpy as np

def load_data(path):
    """Return real public counts with batch-first shapes, without pickle."""
    with np.load(Path(path),allow_pickle=False) as z:
        result={k:z[k].copy() for k in z.files}
    expected={'observations':(1,77,55,80,21,2),'lake_calibration':(1,30,30,30,21,2)}
    if set(result)!=set(expected):raise ValueError('Unexpected observation keys')
    for k,shape in expected.items():
        if result[k].shape!=shape or result[k].dtype!=np.uint16:raise ValueError(f'Invalid {k} shape or dtype')
    return result

def subset_indices(subset='ALL'):
    """Published acquisition groups. Bad registered channel1 is input-defined."""
    if subset=='ALL':return np.delete(np.arange(21),1),np.delete(np.arange(21),1)
    if subset=='SIX-NO-TILT':return np.array([0,2,5]),np.array([0,2,5])
    if subset=='SIX-WITH-TILT':return np.array([7,9,11]),np.array([2,11,18])
    raise ValueError('Unknown published acquisition subset')

def calibration_fit(lake):
    """Fit six a+b*cos2theta+c*sin2theta curves; return offset degrees.

    Input XYZPV unsigned counts. Return offset,21x2 float32 means,6x3
    coefficients,6 offsets from the original1000-point0..180degree maximizer.
    """
    counts=np.asarray(lake,dtype=np.float32)
    if counts.shape!=(30,30,30,21,2):raise ValueError('Invalid lake shape')
    angles=np.deg2rad([0,45,60,90,120,135,180])
    A=np.column_stack([np.ones(7),np.cos(2*angles),np.sin(2*angles)])
    grid=np.linspace(0,180,1000);t=np.deg2rad(grid)
    offsets=[];coefficients=[]
    for view in range(2):
        for tilt in range(3):
            means=counts[...,7*tilt:7*(tilt+1),view].mean(axis=(0,1,2))
            abc=np.linalg.lstsq(A,means,rcond=None)[0]
            fitted=abc[0]+abc[1]*np.cos(2*t)+abc[2]*np.sin(2*t)
            offsets.append(grid[np.argmax(fitted)]);coefficients.append(abc)
    return float(np.mean(offsets)),counts.mean(axis=(0,1,2)),np.asarray(coefficients),np.asarray(offsets)

def polarization_vectors(ia,ib,offset):
    """Return normalized direction vectors(view,selected_state,XYZ)."""
    theta=np.deg2rad(np.array([0,45,60,90,120,135,180])-offset)
    p=np.zeros((2,21,3))
    for k,(a,b) in enumerate([(0,0),(8.9,7.7),(-9.2,-6.5)]):
        c,s=np.cos(theta),np.sin(theta);a,b=np.deg2rad([a,b])
        p[0,7*k:7*(k+1)]=np.stack([np.sin(a)*c,np.cos(a)*c,-s],axis=-1)
        p[1,7*k:7*(k+1)]=np.stack([-s,np.cos(b)*c,np.sin(b)*c],axis=-1)
    p=np.stack([p[0,ia],p[1,ib]])
    return p/np.linalg.norm(p,axis=-1,keepdims=True)

def prepare_observations(raw,subset='ALL'):
    """Return selected float32 normalized counts, polarization and calibration."""
    offset,means,abc,offsets=calibration_fit(raw['lake_calibration'][0])
    ia,ib=subset_indices(subset)
    counts=(raw['observations'][0]/np.iinfo(np.uint16).max).astype(np.float32)
    observations=np.stack([counts[...,ia,0],counts[...,ib,1]],axis=-1)
    cal=np.stack([means[ia,0],means[ib,1]])
    return {'observations':observations,'pols':polarization_vectors(ia,ib,offset),'calibration_averages':cal,'offset':offset,'calibration_coefficients':abc,'calibration_offsets':offsets,'all_calibration_averages':means}
