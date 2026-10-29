"""Constrained Levenberg--Marquardt; explicit local MIT Gpufit extraction."""
import time
import numpy as np
from .physics_model import equations,fma,initial_guess,physical_fields
def gauss_jordan(a,b):
    aug=np.concatenate((a,b[:,:,None]),axis=-1)
    n=len(a);ix=np.arange(n);pivots=[]
    for row in range(6):
        order=np.array([0,4,2,1,5,3])
        pivot=order[np.argmax(abs(aug[:,row,order]),axis=1)];pivots.append(pivot)
        aug[:,row,:]/=aug[ix,row,pivot,None]
        subtract=aug[:,row:row+1,:]*aug[ix,:,pivot][:,:,None]
        subtract[:,row]=0
        aug-=subtract
    out=np.empty_like(b)
    for row,pivot in enumerate(pivots):out[ix,pivot]=aug[:,row,6]
    return out
def fit(x,data,initial,steps=1000,dtype=np.float32,capture=None):
    p=np.asarray(initial,dtype=dtype).copy();data=np.asarray(data,dtype=dtype)
    lo=np.array([2,.0001,.003,.003,.003,-np.inf],dtype=dtype)
    hi=np.array([3.1,.005,np.inf,np.inf,np.inf,np.inf],dtype=dtype)
    p=np.clip(p,lo,hi);obj,g,h=equations(x,p,data)
    scale=np.zeros((len(p),6),dtype=dtype);lam=np.full(len(p),.001,dtype=dtype)
    finished=np.zeros(len(p),bool);iterations=np.zeros(len(p),int);failed=np.zeros(len(p),bool)
    for step in range(steps):
        idx=np.flatnonzero(~finished)
        if not len(idx):break
        diag=np.arange(6);undo=idx[failed[idx]]
        h[undo[:,None],diag,diag]-=scale[undo]*lam[undo,None]/10
        scale[idx]=np.maximum(scale[idx],np.diagonal(h[idx],axis1=1,axis2=2))
        h[idx[:,None],diag,diag]=fma(scale[idx],lam[idx,None],h[idx[:,None],diag,diag])
        a=h[idx].copy()
        delta=gauss_jordan(a,g[idx])
        candidate=np.clip(p[idx]+delta,lo,hi)
        newobj,newg,newh=equations(x,candidate,data[idx])
        converged=abs(newobj-obj[idx])<1e-10*np.maximum(1,newobj)
        accept=newobj<obj[idx]
        use=idx[accept];p[use]=candidate[accept];obj[use]=newobj[accept]
        g[use]=newg[accept];h[use]=newh[accept]
        failed[idx]=~accept
        lam[idx]=np.where(accept,lam[idx]*.1,lam[idx]*10)
        iterations[idx]=step+1;finished[idx[converged]]=True
        if capture is not None:capture(step+1,p.copy())
    return p,obj,iterations

def reconstruct(observations,max_iterations=10000,chunk_size=2048):
    """Fit every spectrum; return four fields and diagnostics without file writes."""
    start=time.perf_counter();data=observations['spectra'][0]
    height,width=data.shape[2:4];count=height*width;data=data.reshape(2,2,count,51)
    frequency=observations['frequency_hz'][0]/np.float32(1e9)
    parameters=np.empty((2,2,count,6),dtype='float32')
    sse=np.empty((2,2,count),dtype='float32');iterations=np.empty((2,2,count),dtype='int32')
    for pol in range(2):
        for window in range(2):
            for first in range(0,count,chunk_size):
                sl=slice(first,min(first+chunk_size,count))
                guess=initial_guess(data[pol,window,sl],frequency[window])
                parameters[pol,window,sl],sse[pol,window,sl],iterations[pol,window,sl]=fit(frequency[window],data[pol,window,sl],guess,max_iterations)
    return dict(reconstruction=physical_fields(parameters,height,width),parameters=parameters,sse=sse,iterations=iterations,seconds=time.perf_counter()-start)
