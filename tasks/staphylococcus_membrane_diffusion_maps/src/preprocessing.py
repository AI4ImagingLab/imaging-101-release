"""Explicit intensity conditioning, variance-preserving bleaching and multi-tau ACF.

Selected pyimfcs algorithms: Aurelien Barbotin, GPL-3.0.
Multi-tau register extraction: multipletau authors, BSD-3-Clause; see NOTICE.
"""
import warnings
import numpy as np
from scipy.optimize import curve_fit

DT=0.0013096382608637214
PIXEL_SIZE=.16
FIRST_FRAME=1500
BINNING=2

def prepare_stack(frames,first_frame=FIRST_FRAME):
    """Return retained uint16 stack after global selected-stack minimum subtraction."""
    frames=np.asarray(frames)
    if frames.ndim!=4 or frames.shape[0]!=1 or frames.dtype!=np.uint16:
        raise ValueError('Expected batch-first uint16 movie')
    selected=frames[0,first_frame:]
    if len(selected)<1000:raise ValueError('Too few stationary frames')
    return selected-selected.min()

def bin_labels(labels,binning=BINNING):
    """Author rule: background plus one cell is accepted; mixed cells are excluded."""
    labels=np.asarray(labels)
    if labels.ndim==3:
        if labels.shape[0]!=1:raise ValueError('Exactly one label image required')
        labels=labels[0]
    if labels.ndim!=2 or binning<1:raise ValueError('Invalid labels/binning')
    h,w=labels.shape; result=np.zeros((h//binning,w//binning),np.uint8)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            block=labels[y*binning:(y+1)*binning,x*binning:(x+1)*binning]
            cells=np.unique(block[block>0])
            if cells.size==1:result[y,x]=cells[0]
    return result

def bleaching_envelope(frame,amplitude,tau,fraction,tau2,offset):
    """Frame-number double exponential; tau and tau2 are in frames, not seconds."""
    return amplitude*((1-fraction)*np.exp(-frame/tau)+fraction*np.exp(-frame/tau2))+offset

def correct_bleaching(trace,return_details=False):
    """Fit original 500-frame means, then correct every original frame.

    Preserve source bounded TRF defaults, p0, half-frame fit convention and
    integer-frame evaluation. Source fallback returns the input after warning.
    """
    trace=np.asarray(trace,dtype=np.float64)
    if trace.ndim!=1 or trace.size<1000 or not np.isfinite(trace).all() or trace.max()<=0:
        raise ValueError('Bleaching requires at least1000 finite positive-signal frames')
    scale=trace.max();normalized=trace/scale;nblocks=trace.size//500
    means=normalized[:nblocks*500].reshape(-1,500).mean(axis=1)
    centers=(np.arange(nblocks)+.5)*500
    initial=(means.max(),centers.max()//10,.5,centers.max()//2,means.min())
    bounds=((0,1,0,1,0),(2*means.max(),centers.max()*10,1,centers.max()*10,1))
    try:
        parameters,_=curve_fit(bleaching_envelope,centers,means,p0=initial,bounds=bounds)
    except Exception as exc:
        warnings.warn('Original bleaching fallback: '+str(exc),RuntimeWarning)
        result=trace.copy(); details={'success':False,'error':str(exc)}
    else:
        envelope=bleaching_envelope(np.arange(trace.size),*parameters)
        root=np.sqrt(envelope/envelope[0])
        result=(normalized/root+envelope[0]*(1-root))*scale
        details={'success':True,'parameters':parameters,'envelope':envelope*scale,'block_centers':centers,'block_means':means*scale}
    return (result,details) if return_details else result

def correlate_trace(trace,dt=DT):
    """Normalized unbiased multi-tau covariance, m16, pair means, zero lag omitted.

    Explicitly retain original final-register truncation and original-mean
    normalization; this is not an FFT correlator with different lag weights.
    """
    trace=np.array(trace,dtype=np.float64,copy=True)
    if trace.ndim!=1 or len(trace)<32 or not np.isfinite(trace).all() or trace.mean()==0 or dt<=0:
        raise ValueError('Invalid correlation trace or sampling interval')
    m=16;length=len(trace);levels=int(np.floor(np.log2(length/m)))
    g=np.zeros((m+levels*(m//2)+1,2));counts=np.zeros(len(g))
    mean=np.average(trace); trace-=mean
    for lag in range(m+1):
        g[lag]=(dt*lag,np.sum(trace[:length-lag]*trace[lag:]));counts[lag]=length-lag
    if length%2:trace=trace[:-1]
    trace=(trace[::2]+trace[1::2])/2;length//=2
    for level in range(1,levels+1):
        for n in range(1,m//2+1):
            lag=n+m//2;index=m+n+(level-1)*m//2
            if len(trace[:length-lag])==0:
                g=g[:index-1];counts=counts[:index-1];break
            g[index]=(dt*lag*2**level,np.sum(trace[:length-lag]*trace[lag:]))
            counts[index]=length-lag
        if length%2:trace=trace[:-1]
        trace=(trace[::2]+trace[1::2])/2;length//=2
    g[:,1]/=mean**2*counts
    return g[1:]
