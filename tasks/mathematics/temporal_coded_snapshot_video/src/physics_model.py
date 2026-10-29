"""Explicit calibrated temporal integration, adjoint and finite-boundary differences.

Mathematics: Yuan, GAP-TV2016; Yuan et al., PnP-SCI2020. Independent local
implementation of the selected author algorithms; upstream source remains private.
"""
import numpy as np
SUBFRAMES=14
BURSTS=15

def forward(cube,mask):
    """Time-first(T,H,W) -> integrated(H,W), native increasing-mask sum order."""
    result=np.zeros(cube.shape[1:],dtype=np.float64)
    for frame,code in zip(cube,mask):result+=frame*code
    return result

def adjoint(image,mask):
    """Integrated(H,W) -> weighted stack(T,H,W); no inverse normalization."""
    return mask*image[None]

def difference(cube,axis):
    """Interior forward differences along spatial axis1(y) or2(x)."""
    if axis not in [1,2]:raise ValueError('Only per-frame spatial TV is selected')
    return np.diff(cube,axis=axis)

def difference_adjoint(dual,axis):
    """Negative divergence with native zero-flux boundaries, no wraparound."""
    if axis not in [1,2]:raise ValueError('Only per-frame spatial TV is selected')
    first=np.take(dual,[0],axis=axis);last=np.take(dual,[-1],axis=axis)
    return np.concatenate([-first,-np.diff(dual,axis=axis),last],axis=axis)

def chronological_cube(cube,burst_index):
    """Invert the sweep on zero-based odd bursts (native downup, iframe1)."""
    return cube if burst_index%2==0 else cube[::-1]

def observed_residuals(movie,observations,mask):
    """One relative observation-L2 residual per complete chronological burst."""
    result=[]
    for k,y in enumerate(observations):
        cube=chronological_cube(movie[k*len(mask):(k+1)*len(mask)],k)
        result.append(float(np.linalg.norm(forward(cube,mask)-y)/np.linalg.norm(y)))
    return np.asarray(result)
