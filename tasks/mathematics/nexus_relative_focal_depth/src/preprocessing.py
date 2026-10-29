"""GPL-2.0 extraction of Haarbach/Mack/Schlaffer VDFF preprocessing.

The copyright attribution and exact upstream mapping are in NOTICE and private
validation. Images are native OpenCV-decoded BGR; no domain package is imported.
"""
import numpy as np
from pathlib import Path


def load_observations(path=None):
    """Load only public native observations; return uint8[19,1080,1920,3]."""
    if path is None:path=Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'observations'}:raise ValueError('Unexpected public observation keys')
        image=data['observations']
    if image.shape!=(1,19,1080,1920,3) or image.dtype!=np.uint8:
        raise ValueError('Expected native batch-first uint8 BGR observations')
    return image[0]


def fused_multiply_add(a,b,c):
    """Approximate source binary32 FMA via binary64 product/sum then cast.

    Two binary32 factors have an exact binary64 product. Their sum with c
    may still round in binary64; a rare double-rounding tie can differ from
    true FMA by one binary32 ULP. This is not a general exact FMA guarantee.
    Actual native stages/full runs are checked against independent source
    fixtures and the predeclared source-compiler numerical tolerance.
    """
    return (np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64)+np.asarray(c,dtype=np.float64)).astype(np.float32)


def modified_laplacian(observations):
    """BGR uint8[F,H,W,3] -> float32[F,H,W] channel-summed MLAP.

    Replicate boundary; pixel conversion uses the original binary32 reciprocal
    1/255, preserving the original multiply rather than a different division.
    """
    observations=np.asarray(observations)
    if observations.dtype!=np.uint8 or observations.ndim!=4 or observations.shape[-1]!=3:
        raise ValueError('Expected uint8 focal-frame,row,column,BGR array')
    sharp=np.zeros(observations.shape[:-1],np.float32)
    for k,frame in enumerate(observations):
        image=frame.astype(np.float32)*np.float32(1/255)
        padded=np.pad(image,((1,1),(1,1),(0,0)),mode='edge')
        for channel in range(3):
            u=image[...,channel]
            xx=(padded[1:-1,:-2,channel]-np.float32(2)*u)+padded[1:-1,2:,channel]
            yy=(padded[:-2,1:-1,channel]-np.float32(2)*u)+padded[2:,1:-1,channel]
            sharp[k]+=np.abs(xx)+np.abs(yy)
    return sharp


def box_replicate_cuda(image,width):
    """Original square-convolution tap order, with approximate binary32 FMA.

    Separating a box convolution changes its float32 accumulation and may
    change a tied discrete argmax. Preserve the original tap order instead.
    """
    r=width//2;p=np.pad(image,((r,r),(r,r)),mode='edge')
    out=np.zeros_like(image,dtype=np.float32);h,w=image.shape
    weight=np.float32(1/(width*width))
    for x in range(width):
        for y in range(width):
            out=fused_multiply_add(p[y:y+h,x:x+w],weight,out)
    return out


def initial_focal_coordinate(sharpness,index_offset=1):
    """Original15x15 MLAP argmax ->21x21 index mean -> shifted relative node.

    index_offset=1 explicitly preserves the source port's one-frame negative
    initializer shift. index_offset=0 is a separately labelled diagnostic.
    This is initialization, never an instrument distance calibration.
    """
    f,h,w=sharpness.shape
    best=np.zeros((h,w),np.float32);index=np.zeros((h,w),np.float32)
    for k in range(f):
        smoothed=box_replicate_cuda(sharpness[k],15)
        take=smoothed>best
        index[take]=k;best[take]=smoothed[take]
    smoothed_index=box_replicate_cuda(index,21)
    return np.float32(20)*((smoothed_index-np.float32(index_offset))/np.float32(f-1)-np.float32(.5))


def lu_inverse_float32(matrix):
    """Small partial-pivoted binary32 Gaussian elimination, no opaque inverse.

    Operation order follows the original OpenCV3.4 general small-matrix LU,
    needed for the ill-conditioned degree8 monomial normal equations.
    """
    a=np.array(matrix,dtype=np.float32,copy=True);n=len(a)
    b=np.eye(n,dtype=np.float32)
    for i in range(n):
        pivot=i+int(np.argmax(np.abs(a[i:,i])))
        if abs(a[pivot,i])<10*np.finfo(np.float32).eps:raise ValueError('Singular normal matrix')
        if pivot!=i:
            a[[i,pivot],i:]=a[[pivot,i],i:];b[[i,pivot]]=b[[pivot,i]]
        d=np.float32(-1)/a[i,i]
        for j in range(i+1,n):
            alpha=np.float32(a[j,i]*d)
            a[j,i+1:]+=alpha*a[i,i+1:]
            b[j]+=alpha*b[i]
    for i in range(n-1,-1,-1):
        for j in range(n):
            s=b[i,j]
            for k in range(i+1,n):s=np.float32(s-np.float32(a[i,k]*b[k,j]))
            b[i,j]=np.float32(s/a[i,i])
    return b


def polynomial_operator(frames=19,degree=8):
    """Descending float32 monomials and source normal-equation left inverse.

    Source GEMM accumulates in binary64 then serializes binary32. LU uses
    binary32; replacing this by a float64/SVD fit changes the selected method.
    """
    node=np.float32(20)*(np.arange(frames,dtype=np.float32)/np.float32(frames-1)-np.float32(.5))
    X=np.ones((frames,degree+1),np.float32)
    current=node.copy()
    for j in range(degree-1,-1,-1):X[:,j]=current;current*=node
    normal=(X.astype(np.float64).T@X.astype(np.float64)).astype(np.float32)
    inverse=lu_inverse_float32(normal)
    P=(inverse.astype(np.float64)@X.T.astype(np.float64)).astype(np.float32)
    return X,P


def fit_focus_polynomials(sharpness,degree=8,denominator_regularizer=.3):
    """Fit normalized focus curves and return their NEGATIVE derivatives.

    Returns positive contrast coefficients[degree+1,H,W] and negative
    derivative coefficients[degree,H,W], both binary32, descending powers.
    """
    _,P=polynomial_operator(len(sharpness),degree)
    scaling=np.float32(1)/(np.max(sharpness,axis=0)+np.float32(denominator_regularizer))
    normalized=np.asarray(sharpness,np.float32)*scaling
    coefficients=np.zeros((degree+1,*sharpness.shape[1:]),np.float32)
    for i,row in enumerate(P):
        for j,weight in enumerate(row):coefficients[i]=fused_multiply_add(weight,normalized[j],coefficients[i])
    derivative=-(coefficients[:-1]*np.arange(degree,0,-1,dtype=np.float32)[:,None,None])
    return coefficients,derivative
