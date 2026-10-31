"""Observation-only SEARCH support and float32 spectral moments.

Source conventions are from BBarolo, GPL-2.0-or-later. Detection is not a
target-derived calibration: it is recomputed solely from the public cube.
"""
import numpy as np
from scipy import ndimage

VELOCITY_START_KM_S=289.610969
VELOCITY_STEP_KM_S=-5.152860000000001
PIXEL_ARCSEC=16.00000038756


def _components(array):
    """Chebyshev sky-adjacency 1 and spectral separation <=2 channels."""
    labels,count=ndimage.label(array,structure=np.ones((3,3,3),dtype=bool))
    parent=np.arange(count+1,dtype=np.int32)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for dy in [-1,0,1]:
        ya=slice(max(0,dy),min(array.shape[1],array.shape[1]+dy))
        yb=slice(max(0,-dy),min(array.shape[1],array.shape[1]-dy))
        for dx in [-1,0,1]:
            xa=slice(max(0,dx),min(array.shape[2],array.shape[2]+dx))
            xb=slice(max(0,-dx),min(array.shape[2],array.shape[2]-dx))
            a=labels[:-2,ya,xa];b=labels[2:,yb,xb]
            valid=(a!=0)&(b!=0)&(a!=b)
            pairs=np.unique(np.stack((a[valid],b[valid]),axis=1),axis=0)
            for i,j in pairs:
                ri,rj=root(i),root(j)
                if ri!=rj:parent[rj]=ri
    lookup=np.array([root(i) for i in range(count+1)])
    return lookup[labels]


def search_mask(cube):
    """Robust 4-sigma source search grown to 2.5-sigma; largest object.

    Blank zero observations are excluded from the median/MAD statistics.
    Source minimum support is ceil(Gaussian beam area)=16 sky pixels and
    at least two channels / 32 voxels, before and after source merging.
    """
    cube=np.asarray(cube,dtype=np.float32)
    valid=np.isfinite(cube)&(cube!=0)
    values=cube[valid]
    middle=np.float32(np.median(values))
    mad=np.float32(np.median(np.abs(values-middle)))
    spread=np.float32(mad/np.float32(.6744888))
    threshold=np.float32(middle+np.float32(4)*spread)
    growth=np.float32(middle+np.float32(2.5)*spread)
    high=_components(cube>threshold)
    high_ids=np.unique(high[high!=0])
    selected=[]
    for identifier in high_ids:
        z,y,x=np.where(high==identifier)
        if z.size<32 or np.unique(y*cube.shape[2]+x).size<16:continue
        channels=np.unique(z)
        if not np.any(np.diff(channels)==1):continue
        selected.append(identifier)
    seed=np.isin(high,selected)&(high!=0)
    low=_components(cube>growth)
    candidates=np.unique(low[seed]);candidates=candidates[candidates!=0]
    if not candidates.size:
        raise ValueError('No emission source satisfies the declared SEARCH support')
    sizes=np.bincount(low.ravel())
    identifier=candidates[np.argmax(sizes[candidates])]
    return low==identifier,dict(median=float(middle),mad=float(mad),spread=float(spread),threshold=float(threshold),growth_threshold=float(growth))


def first_moment(cube, mask):
    """Source masked intensity-weighted LOS velocity, km/s; missing is NaN."""
    cube=np.asarray(cube,dtype=np.float32)
    numerator=np.zeros(cube.shape[1:],dtype=np.float32)
    denominator=np.zeros_like(numerator)
    velocity=VELOCITY_START_KM_S+VELOCITY_STEP_KM_S*np.arange(cube.shape[0])
    for z in range(cube.shape[0]):
        # Source numerator is accumulated in double then assigned to float32;
        # denominator adds a float32 product to a float32 accumulator.
        numerator=(numerator.astype(float)+cube[z].astype(float)*velocity[z]*mask[z]).astype(np.float32)
        denominator+=cube[z]*mask[z]
    out=np.full_like(numerator,np.nan)
    np.divide(numerator,denominator,out=out,where=denominator!=0)
    return out
