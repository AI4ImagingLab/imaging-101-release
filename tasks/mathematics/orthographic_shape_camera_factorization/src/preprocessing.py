"""Complete multi-frame point-track input and centring conventions."""
import numpy as np

def load_observations(path):
    """Return real point tracks[51,491,2], image-coordinate pixel units."""
    with np.load(path,allow_pickle=False) as z:a=z['tracks']
    if a.shape!=(1,51,491,2) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Expected finite float64 tracks[1,51,491,2]')
    return a[0].copy()

def centre_and_scale(tracks):
    """Return interleaved centred/scaled W[2F,P],centroids[2F,1],scale."""
    W=tracks.transpose(0,2,1).reshape(2*len(tracks),tracks.shape[1])
    translation=W@np.ones((W.shape[1],1))/W.shape[1]
    centred=W-translation
    scale=float(np.max(np.abs(centred[::2]))+np.max(np.abs(centred[1::2])))
    if scale<=0:raise ValueError('Degenerate point tracks')
    return centred/scale,translation,scale
