import numpy as np
from scipy import sparse
from src.physics_model import fluorescence_forward,fluorescence_adjoint,get_noise_fft
from src.solvers import normalize,positive_lars,constrained_oasisAR1
from src.preprocessing import apply_shift_iteration

def test_fluorescence_spatial_adjoint():
    rng=np.random.RandomState(9);A=rng.randn(17,3);b=rng.randn(17,2);C=rng.randn(3,23);f=rng.randn(2,23);R=rng.randn(17,23)
    dA,db=fluorescence_adjoint(R,C,f)
    lhs=np.sum(fluorescence_forward(A,C,b,f)*R)
    rhs=np.sum(A*dA)+np.sum(b*db)
    np.testing.assert_allclose(lhs,rhs,rtol=1e-13,atol=1e-12)

def test_normalization_preserves_movie():
    rng=np.random.RandomState(3);r={'A':sparse.csc_matrix(rng.rand(9,2)),'C':rng.rand(2,31),'b':rng.rand(9,2),'f':rng.rand(2,31)}
    before=fluorescence_forward(r['A'],r['C'],r['b'],r['f']);normalize(r)
    after=fluorescence_forward(r['A'],r['C'],r['b'],r['f'])
    np.testing.assert_allclose(before,after,rtol=2e-7,atol=2e-7)

def test_noise_psd_white_scale():
    rng=np.random.RandomState(123);Y=(rng.randn(64,3000)*3).astype(np.float32)
    sn,_=get_noise_fft(Y,noise_method='mean')
    assert abs(np.mean(sn)-3)<.05

def test_motion_constant_and_integer_direction():
    flat=np.ones((19,19),np.float32)*7
    np.testing.assert_allclose(apply_shift_iteration(flat,(2,-3),border_nan='copy'),flat)
    image=np.zeros((19,19),np.float32);image[9,9]=1
    moved=apply_shift_iteration(image,(2,-3),border_nan='copy')
    assert np.unravel_index(np.argmax(moved),moved.shape)==(11,6)

def test_ar_recurrence_and_noise_constraint():
    rng=np.random.RandomState(17);g=.9;s=np.zeros(100);s[[15,45,75]]=[4,6,3];c=np.zeros(100)
    for t in range(1,100):c[t]=g*c[t-1]+s[t]
    y=(c+rng.randn(100)*.1).astype(np.float32)
    calcium,events,baseline,_,_=constrained_oasisAR1(y,g,.2,optimize_b=True,b_nonneg=False)
    np.testing.assert_allclose(events[1:],calcium[1:]-g*calcium[:-1],atol=1e-6)
    assert events.min()>-1e-5
    assert np.mean((y-baseline-calcium)**2)<=.0401

def test_positive_lars_orthogonal_direction():
    X=np.array([[1,0],[-1,0],[0,1],[0,-1]],float);y=X@np.array([3.,1.])
    coef=positive_lars(X,y,.1)
    assert np.all(coef>=0)
    assert coef[0]>coef[1]>0
