"""Explicit finite-dictionary inverse and author fixed-window comparison rules."""
import time
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter
from .preprocessing import normalize_observations
from .physics_model import calibration_rays

def decode_rays(colors,depths,query,mask):
    """Exhaustive RGB squared distance, earliest minimum, then Euclidean norm.

    Preserve float32 sequential channel accumulation. Unsupported rays return
    depth-1/index0/lossInf internally; only finite depth is submitted.
    """
    colors=np.asarray(colors,dtype=np.float32);depths=np.asarray(depths,dtype=np.float32);query=np.asarray(query,dtype=np.float32);mask=np.asarray(mask,dtype=bool)
    n,z,c=colors.shape
    if c!=3 or depths.shape!=(n,z) or query.shape!=(n,3) or mask.shape!=(n,):raise ValueError('Incompatible ray-array shapes')
    d=np.full(n,-1,dtype=np.float32);idx=np.zeros(n,dtype=np.int32);loss=np.full(n,np.inf,dtype=np.float32)
    selected=np.flatnonzero(mask)
    if not len(selected):return d,idx,loss
    q=query[selected];a=colors[selected]
    diff=a[:,:,0]-q[:,None,0];distance=diff*diff
    for channel in [1,2]:
        diff=a[:,:,channel]-q[:,None,channel];distance+=diff*diff
    chosen=distance.argmin(axis=1)
    idx[selected]=chosen;d[selected]=depths[selected,chosen];loss[selected]=np.sqrt(distance[np.arange(len(selected)),chosen])
    return d,idx,loss

def solve(raw,block_size=2048,rank=25,query_override=None):
    """Actively decode all complete input fields; temporary dictionaries bounded.

    Returns reconstruction(F,H,W), index(F,H,W), loss(F,H,W), normalized,
    mask and actual elapsed seconds. Only acquisition-derived unsupported rays
    are skipped; no result/residual clipping or geometric filtering occurs.
    """
    started=time.perf_counter();q,mask=normalize_observations(raw['pattern'][0],raw['white'][0])
    if query_override is not None:
        q=np.asarray(query_override,dtype=np.float32)
        if q.shape!=raw['pattern'][0].shape or not np.isfinite(q).all():raise ValueError('Invalid review-only normalized observation')
    if not isinstance(block_size,int) or block_size<1:raise ValueError('Positive block size required')
    frames,h,w=q.shape[:3];d=np.full((frames,h*w),-1,dtype=np.float32);index=np.full((frames,h*w),-1,dtype=np.int32);loss=np.full((frames,h*w),np.inf,dtype=np.float32)
    flatq=q.reshape(frames,-1,3);flatmask=mask.reshape(frames,-1);rows=np.flatnonzero(flatmask.any(axis=0))
    for start in range(0,len(rows),block_size):
        r=rows[start:start+block_size];color,depth=calibration_rays(raw,r,rank)
        for frame in range(frames):
            di,ii,li=decode_rays(color,depth,flatq[frame,r],flatmask[frame,r]);ii[~flatmask[frame,r]]=-1
            d[frame,r]=di;index[frame,r]=ii;loss[frame,r]=li
    shape=(frames,h,w)
    return dict(reconstruction=d.reshape(shape),index=index.reshape(shape),loss=loss.reshape(shape),normalized=q,mask=mask,seconds=time.perf_counter()-started)

def _window_decode(raw,q,mask,rows,previous,delta,block_size=2048):
    previous=np.asarray(previous);starts=np.clip(previous.astype(np.int16)-delta,0,749-2*delta).astype(np.uint16)
    depth=np.full(len(rows),-1,dtype=np.float32);relative=np.full(len(rows),-1,dtype=np.int32);loss=np.full(len(rows),np.inf,dtype=np.float32)
    positions=np.flatnonzero(mask)
    for offset in range(0,len(positions),block_size):
        chosen=positions[offset:offset+block_size];c,d=calibration_rays(raw,rows[chosen]);j=starts[chosen,None]+np.arange(2*delta)[None]
        cc=np.take_along_axis(c,j[:,:,None],axis=1);dd=np.take_along_axis(d,j,axis=1)
        di,ii,li=decode_rays(cc,dd,q[chosen],np.ones(len(chosen),dtype=bool));depth[chosen]=di;relative[chosen]=ii;loss[chosen]=li
    return depth,relative,loss,starts

def _nearest(array,condition):
    indices=distance_transform_edt(condition,return_distances=False,return_indices=True)
    return array[tuple(indices)]

def coarse_to_fine(raw,q,mask,ks=(8,4,2,1),deltas=(374,125,50,25),epsilon=.1):
    """Qualified source helper: fixed windows, not published adaptive fallback.

    Returns the source's final *relative* indices; they cannot serve as global
    temporal priors. Source int16 Gaussian filtering and endpoint exclusion are
    preserved, while the independently proven depth-gather broadcast is fixed.
    """
    h,w=mask.shape;previous=np.zeros((h,w),dtype=np.int16);loss=np.full((h,w),np.inf,dtype=np.float32);grid=np.arange(h*w).reshape(h,w)
    for level,k in enumerate(ks[:-1]):
        rr=grid[::k,::k];di,ii,li,start=_window_decode(raw,q[::k,::k].reshape(-1,3),mask[::k,::k].ravel(),rr.ravel(),previous[::k,::k].ravel(),deltas[level])
        previous[::k,::k]=(ii+start).reshape(rr.shape);loss[::k,::k]=li.reshape(rr.shape)
        previous=_nearest(previous,previous<=0);previous=_nearest(previous,loss>epsilon)
        previous=gaussian_filter(previous,sigma=k//ks[level+1],mode='nearest')
    d,i,l,start=_window_decode(raw,q.reshape(-1,3),mask.ravel(),grid.ravel(),previous.ravel(),deltas[-1])
    return dict(reconstruction=d.reshape(h,w),relative_index=i.reshape(h,w),loss=l.reshape(h,w))

def temporal_continuation(raw,q,mask,previous_index,delta=25):
    """Actual source tc_lut API with direct previous global index, no fallback."""
    shape=mask.shape;d,i,l,start=_window_decode(raw,q.reshape(-1,3),mask.ravel(),np.arange(mask.size),np.asarray(previous_index).ravel(),delta)
    return dict(reconstruction=d.reshape(shape),index=(i+start).reshape(shape),loss=l.reshape(shape))
