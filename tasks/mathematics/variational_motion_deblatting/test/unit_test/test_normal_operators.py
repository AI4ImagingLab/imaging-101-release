"""Compare solver normal operators against explicit direct-convolution matrices."""
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from src.solvers import Params, estimateH, estimateFM


def direct_convolve(a,b):
    out=np.zeros_like(a)
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            out+=a[row,col]*np.roll(b,(row,col),axis=(0,1))
    return out


def embed(x,shape):
    result=np.zeros(shape)
    result[:x.shape[0],:x.shape[1]]=x
    return np.roll(result,tuple(-(np.array(x.shape)//2)),axis=(0,1))


def model(b,h,f,m):
    # Independent loops, no FFT, and an explicit object-center shift.
    h_m=direct_convolve(h,embed(m,h.shape))
    return np.stack([direct_convolve(h,embed(f[:,:,c],h.shape))+(1-h_m)*b[:,:,c] for c in range(3)],axis=2)


def test_kernel_normal_operator_against_explicit_matrix(monkeypatch):
    rng=np.random.default_rng(8);b=rng.random((7,9,3));f=rng.random((3,3,3));m=rng.random((3,3))
    identity=np.eye(63)
    k=np.column_stack([(model(b,x.reshape(7,9),f,m)-b).ravel() for x in identity])
    params=Params();params.maxiter=1;params.verbose=False;params.visualize=False
    expected=k.T@k+params.beta_h*np.eye(63)
    real_cg=scipy.sparse.linalg.cg
    checked=[]
    def audit(a,rhs,*args,**kwargs):
        actual=np.column_stack([a@x for x in identity])
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
        assert np.linalg.eigvalsh(actual).min()>0
        checked.append(True)
        return real_cg(a,rhs,*args,**kwargs)
    monkeypatch.setattr(scipy.sparse.linalg,'cg',audit)
    estimateH(b+.01,b,m,f,params=params)
    assert checked


def test_joint_normal_operator_against_explicit_matrix(monkeypatch):
    rng=np.random.default_rng(9);b=rng.random((7,9,3));h=rng.random((7,9));h/=h.sum()
    identity=np.eye(36)
    def split(x):
        fields=x.reshape(3,3,4,order='F')
        return fields[:,:,:3],fields[:,:,3]
    k=np.column_stack([(model(b,h,*split(x))-b).ravel() for x in identity])
    def independent_gradient(x):
        array=x.reshape(3,3,order='F');padded=np.pad(array,1)
        dx=np.zeros((4,4));dy=np.zeros((4,4))
        dx[:3,:]=np.diff(padded[1:-1,:],axis=1)
        dy[:,:3]=np.diff(padded[:,1:-1],axis=0)
        return np.r_[dx.ravel(),dy.ravel()]
    d=np.column_stack([independent_gradient(x) for x in np.eye(9)])
    params=Params();params.maxiter=1;params.lambda_R=0;params.verbose=False;params.visualize=False
    expected=k.T@k+params.beta_f*sparse.block_diag([d.T@d]*4).toarray()+params.beta_fm*np.eye(36)
    real_cg=scipy.sparse.linalg.cg;checked=[]
    def audit(a,rhs,*args,**kwargs):
        actual=np.column_stack([a@x for x in identity])
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
        assert np.linalg.eigvalsh(actual).min()>0
        checked.append(True)
        return real_cg(a,rhs,*args,**kwargs)
    monkeypatch.setattr(scipy.sparse.linalg,'cg',audit)
    estimateFM(b+.01,b,h,M=np.ones((3,3)),params=params)
    assert checked
