"""Raw energy image loading and the published defective-pixel preprocessing."""
import numpy as np
from scipy.ndimage import median_filter,gaussian_filter
def load_data(path):
    """Return uint32 detector(256,256) keV and float32 binary mask(124,124)."""
    with np.load(path,allow_pickle=False) as z:y=z['detector_energy'];m=z['mask_pattern']
    if y.shape!=(1,256,256) or y.dtype!=np.uint32:raise ValueError('Invalid detector_energy')
    if m.shape!=(1,124,124) or m.dtype!=np.float32 or not np.isin(m,[0,1]).all():raise ValueError('Invalid mask_pattern')
    return y[0],m[0]
def preprocess(raw,sigma=1.,reject_outliers=True):
    """float32(H,W) keV; replace strict outside 1/99 percentiles with 3x3 median.

    Gaussian sigma is in detector pixels, truncate=4, half-sample reflect boundary.
    This preserves upstream float32 intermediate arithmetic and strict inequalities.
    """
    a=np.asarray(raw,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Finite 2D image required')
    if reject_outliers:
        lo,hi=np.percentile(a,[1,99]);bad=np.zeros_like(a,np.float32);bad[(a<lo)|(a>hi)]=1
        a=bad*median_filter(a,size=3)+(1-bad)*a
    return gaussian_filter(a,sigma) if sigma>0 else a.copy()
