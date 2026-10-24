"""Explicit nitrogen-14 ODMR spectra, derivatives and physical units.
Adapted from QDMpy (Mike Volk, MIT) and Gpufit (Bates et al., MIT).
See LICENSE and test/validation.md for source and qualifications.
"""
import numpy as np
def fma(a,b,c):
    """Approximate contracted float32 multiply-add via float64 intermediate.

    Double rounding is possible for arbitrary inputs. This implementation is
    validated on this task's independent captured model values/Jacobians and
    bounded by the frozen physical tolerances; it is not a universal FMA emulator.
    """
    dtype=np.result_type(a,b,c)
    if dtype==np.float32:return (np.asarray(a,dtype=np.float64)*b+c).astype(np.float32)
    return a*b+c
def model_jac(x,p):
    x=np.asarray(x,dtype=p.dtype)[None,:]
    y=np.broadcast_to(1+p[:,5,None],(len(p),x.shape[1])).copy()
    j=np.zeros((*y.shape,6),dtype=p.dtype);j[:,:,5]=1
    w=p[:,1,None];w2=w*w
    terms=[]
    for col,shift in zip((2,3,4),(.002158,0,-.002158)):
        d=x-p[:,0,None]+p.dtype.type(shift);den=fma(d,d,w2); c=p[:,col,None]
        inv=1/den;inv2=1/(den*den)
        y=fma(-c*w2,inv,y)
        j[:,:,col]=-w2*inv
        terms.append((d,inv,inv2,c))
    d0,v0,vv0,c0=terms[0];d1,v1,vv1,c1=terms[1];d2,v2,vv2,c2=terms[2]
    # Preserve the original compiled source expression contraction ordering.
    j[:,:,0]=fma(-2*c0*w2*d0,vv0,(-2*c1*w2*d1)*vv1)
    j[:,:,0]=fma(-(2*c2*w2*d2),vv2,j[:,:,0])
    j[:,:,1]=fma(2*w2*w*c0*v0,v0,-(2*w*c0*v0))
    for d,v,vv,c in terms[1:]:
        j[:,:,1]=fma(2*w2*w*c*v,v,j[:,:,1])
        j[:,:,1]=fma(-(2*w*c),v,j[:,:,1])
    return y,j
def tree_sum(a):
    b=np.zeros((*a.shape[:-1],64),dtype=a.dtype);b[...,:a.shape[-1]]=a
    n=32
    while n:
        b[...,:n]+=b[...,n:2*n];n//=2
    return b[...,0]
def equations(x,p,data):
    y,j=model_jac(x,p);r=data-y
    obj=tree_sum(r*r)
    g=tree_sum((j*r[:,:,None]).swapaxes(1,2))
    h=np.empty((len(p),6,6),dtype=p.dtype)
    for i in range(6):
        for k in range(i+1):
            h[:,i,k]=h[:,k,i]=np.cumsum(j[:,:,i]*j[:,:,k],axis=-1,dtype=np.float64)[:,-1]
    return obj,g,h
def initial_guess(data,frequency):
    cumulative=np.cumsum(data-1,axis=-1)
    cumulative-=np.min(cumulative,axis=-1,keepdims=True)
    cumulative/=np.max(cumulative,axis=-1,keepdims=True)
    center=frequency[np.argmin(abs(cumulative-.5),axis=-1)]
    width=frequency[np.argmin(abs(cumulative-.35),axis=-1)]-frequency[np.argmin(abs(cumulative-.65),axis=-1)]
    contrast=abs((np.max(data,axis=-1)-np.min(data,axis=-1))/np.max(data,axis=-1))
    return np.stack([center,width,contrast,contrast,contrast,np.zeros_like(contrast)],axis=-1).astype('float32')
def physical_fields(parameters,height=300,width=480):
    centers=parameters[...,0].reshape(2,2,height,width)
    signed=(centers[:,1]-centers[:,0])/np.float32(2)/np.float32(28.024e-6)*np.array([-1,1])[:,None,None]
    return np.stack([(signed[0]+signed[1])/2,(signed[0]-signed[1])/2,(centers[0,0]+centers[0,1])/np.float32(2),(centers[1,0]+centers[1,1])/np.float32(2)])[None]
