"""Prepare the exact author's fixed-mask photographic acquisition.

The input RGB uint8 photograph and binary mask are independent author-provided
assets, not generated targets. Convert RGB to .3R+.59G+.11B after /255, exactly
matching the selected original method, then apply the author's fixed mask.
No noise, new random seed or random mask is introduced. Truth is returned to
the maintainer only and must be saved separately from public observations.
"""
import numpy as np
def generate_observations(rgb,mask):
    rgb=np.asarray(rgb);mask=np.asarray(mask)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or mask.shape!=rgb.shape[:2] or not np.isin(mask,[0,1]).all():raise ValueError('Expected uint8 RGB and binary mask')
    x=rgb.astype(np.float64)/255.
    truth=.3*x[:,:,0]+.59*x[:,:,1]+.11*x[:,:,2]
    visible=np.where(mask,truth,0.)
    return dict(observations=visible[None],mask=mask.astype(bool)[None]),dict(reconstruction=truth[None])

