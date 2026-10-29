"""Numeric diagnostics and plots; importing never changes plotting backends."""
import numpy as np
def segment_velocity(positions,frames,ids):
    order=np.lexsort((frames,ids));a,b=order[:-1],order[1:];keep=(ids[a]==ids[b])&(frames[b]==frames[a]+1);a,b=a[keep],b[keep]
    return a,b,positions[b]-positions[a]
def gaussian_smooth_tracks(positions,frames,ids,sigma=1.):
    from scipy.ndimage import gaussian_filter1d
    out=positions.copy();order=np.lexsort((frames,ids));labels=ids[order];cuts=np.r_[0,np.flatnonzero(np.diff(labels))+1,len(order)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        ix=order[a:b]
        if len(ix)>1:out[ix]=gaussian_filter1d(positions[ix],sigma,axis=0,mode='nearest',truncate=3.)
    return out
