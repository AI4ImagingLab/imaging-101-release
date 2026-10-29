"""Explicit numerical conventions; source and license mapping in validation."""
from fractions import Fraction
import numpy as np

def acquisition_log(values):
    """Julia0.7 Tang near-one log (Base.special.log_proc2, MIT).

    All bound E values lie in [0.96703528,0.99835110], within this branch.
    Other arguments use NumPy's general log with potentially different final
    rounding. Julia's recorded FMA_NATIVE predicate is false.
    """
    values=np.asarray(values,dtype=np.float64)
    near=(values>.9394130628134757)&(values<1.0644944589178595)
    f=np.where(near,values,1.)-1.;g=1./(2.+f);u=2.*f*g;v=u*u
    q=u*v*(.08333333333333179+v*(.012500000003771751+v*(.0022321399879194482+v*.0004348877777076146)))
    u1=(u.view(np.uint64)&np.uint64(0xfffffffff8000000)).view(np.float64)
    f1=(f.view(np.uint64)&np.uint64(0xfffffffff8000000)).view(np.float64)
    f2=f-f1
    u2=((2.*(f-u1)-u1*f1)-u1*f2)*g
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(near,u1+(u2+q),np.log(values))

def conv(cp,kep,t):
    kep=np.atleast_1d(kep);n=len(kep);out=np.zeros((n,len(t)))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for j in range(len(t)-1):
            x=kep*(t[j+1]-t[j]);da=(cp[j+1]-cp[j])/x
            e=np.exp(-x);e0=1-e;e1=x-e0
            out[:,j+1]=e*out[:,j]+cp[j]*e0+da*e1
        out=out/kep[:,None]
    return out

def fma_square(a,c):
    """Error-free product/add decomposition for finite nonnegative square sums."""
    with np.errstate(over='ignore',invalid='ignore'):
      a2=a*a
      split=134217729.0*a
      high=split-(split-a);low=a-high
      error=(((high*high-a2)+high*low)+low*high)+low*low
      s=a2+c;z=s-a2
      correction=(a2-(s-z))+(c-z)+error
      result=s+correction
    # The double-double correction may itself round at a final halfway case.
    # Resolve a conservative 32-epsilon neighbourhood with exact rational
    # arithmetic, including the underflow regime, rather than double-rounding.
    with np.errstate(invalid='ignore',over='ignore'):
      half=.5*np.spacing(s)
      ambiguous=np.isfinite(a)&np.isfinite(c)&((abs(abs(correction)-half)<=32*np.finfo(float).eps*abs(half))|((a!=0)&(abs(a)<np.sqrt(np.finfo(float).tiny))))
    result=np.asarray(result)
    if np.any(ambiguous):
      for index in zip(*np.nonzero(ambiguous)):
        av=float(a[index]);cv=float(c[index])
        try:
          aa=Fraction.from_float(av);cc=Fraction.from_float(cv)
          result[index]=float(aa*aa+cc)
        except Exception as exc:
          raise ArithmeticError(f'Exact square-add fallback failed for a={av!r}, c={cv!r}: {exc}') from exc
    return np.where(np.isfinite(result),result,a2+c)

def julia_square_sum(a):
    """Julia0.7 emitted AVX2/FMA four-vector reduction; last axis reduced."""
    a=np.asarray(a)
    if a.shape[-1]>1024:
        mid=(a.shape[-1]+1)//2
        return julia_square_sum(a[...,:mid])+julia_square_sum(a[...,mid:])
    if a.shape[-1]==1:return a[...,0]*a[...,0]
    v=np.zeros(a.shape[:-1]+(4,4))
    v[...,0,0]=a[...,0]*a[...,0]+a[...,1]*a[...,1]
    stop=2+((a.shape[-1]-2)//16)*16
    for i in range(2,stop,16):
        v=fma_square(a[...,i:i+16].reshape(v.shape),v)
    z=((v[...,0,:]+v[...,1,:])+v[...,2,:])+v[...,3,:]
    s=(z[...,0]+z[...,2])+(z[...,1]+z[...,3])
    for i in range(stop,a.shape[-1]):s=fma_square(a[...,i],s)
    return s

def prescott_transpose_product(jt,f):
    # OpenBLAS0.3.2 e8a68ef x86_64/dgemv_t.S, two-column path. Four scalar
    # accumulators preserve its SSE lane/tail grouping without a new backend.
    # Native time count1321 makes the leading dimension odd; .L60 uses two
    # SSE lanes and adds its final odd sample before the horizontal reduction.
    lanes=2 if jt.shape[2]%2 else 4
    v=np.zeros(jt.shape[:2]+(lanes,));end=(jt.shape[2]//lanes)*lanes
    for i in range(0,end,lanes):v+=jt[:,:,i:i+lanes]*f[:,None,i:i+lanes]
    for i in range(end,jt.shape[2]):v[:,:,i%lanes]+=jt[:,:,i]*f[:,None,i]
    return v[:,:,0]+v[:,:,1] if lanes==2 else (v[:,:,0]+v[:,:,2])+(v[:,:,1]+v[:,:,3])

