"""Observed-signal surface support; no target or hidden reference inputs.

Adapted from Smith/Dwork2015 source, GPL-3.0. See LICENSE/NOTICE.
"""
import numpy as np
from scipy.ndimage import maximum_filter,minimum_filter
SHAPE=(512,512)
def load_data(path):
 with np.load(path,allow_pickle=False) as d:
  if set(d.files)!={'oct_signal'}:raise ValueError('Expected only oct_signal')
  a=d['oct_signal']
  if a.shape!=(1,*SHAPE) or a.dtype!=np.dtype('float64') or not np.isfinite(a).all() or np.min(a)<0:raise ValueError('Require finite nonnegativefloat64(1,512,512)')
  return a[0].copy()
def surface_indices(signal):
 """Return number of rows before each native first upward half-range crossing."""
 a=np.asarray(signal);n=a.shape[0];ordered=np.sort(a,axis=0)
 lo=ordered[:int(np.floor(.1*n+.5))].mean(axis=0)
 hi=ordered[int(np.floor(.9*n+.5))-1:].mean(axis=0)
 above=a>((lo+hi)/2);up=np.diff(above.astype(np.int8),axis=0)==1
 return np.where(np.any(up,axis=0),np.argmax(up,axis=0)+1,0)
def tissue_mask(signal):
 """Native surface mask, square10 dilation then square35 erosion.

 Morphological boundaries use zero for dilation/one for erosion; even-square
 origin follows the independently captured Octave image-package convention.
 """
 loc=surface_indices(signal);mask=(np.arange(np.shape(signal)[0])[:,None]>=loc[None,:]).astype(np.uint8)
 mask=maximum_filter(mask,size=10,mode='constant',cval=0,origin=0)
 return minimum_filter(mask,size=35,mode='constant',cval=1).astype(bool)
