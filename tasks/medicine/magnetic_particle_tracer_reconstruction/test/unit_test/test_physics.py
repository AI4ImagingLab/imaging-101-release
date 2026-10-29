import numpy as np
from src.physics_model import forward,adjoint,objective,gradient_real
from src.solvers import kaczmarz,row_norms

def test_complex_adjoint_and_real_gradient():
    rng=np.random.default_rng(42);S=rng.normal(size=(150,7))+1j*rng.normal(size=(150,7));x=rng.normal(size=7);y=rng.normal(size=150)+1j*rng.normal(size=150)
    np.testing.assert_allclose(np.vdot(forward(S,x),y),np.vdot(x,adjoint(S,y)),rtol=1e-13)
    lam=.8;g=gradient_real(S,y,x,lam)
    for j in range(7):
        delta=np.zeros(7);delta[j]=1e-5;fd=(objective(S,y,x+delta,lam)-objective(S,y,x-delta,lam))/2e-5
        assert np.isclose(g[j],fd,rtol=1e-7,atol=1e-6)

def test_one_row_closed_form_conjugation_and_zero_rows():
    S=np.array([[1+2j,3-1j]],dtype=np.complex128);u=np.array([2+3j]);lam=.7
    expected=((u[0]/(np.linalg.norm(S[0])**2+lam))*S[0].conjugate()).real;expected=np.maximum(expected,0)
    np.testing.assert_allclose(kaczmarz(S,u,1,lam),expected,atol=1e-15)
    padded=np.vstack((np.zeros_like(S),S));np.testing.assert_allclose(kaczmarz(padded,np.r_[0j,u],1,lam),expected,atol=1e-15)

def test_augmented_residual_scalar_analytic_solution():
    # An exactly solved one-row augmented system stays at x=a*b/(a²+lambda).
    S=np.array([[2.+0j]]);u=np.array([3.+0j]);lam=5.
    for n in (1,3,10):np.testing.assert_allclose(kaczmarz(S,u,n,lam),[6/9],atol=1e-15)

def test_real_positive_projection_and_native_norm_precision():
    S=np.array([[1+0j,0]],np.complex64);u=np.array([-2+4j]);x=kaczmarz(S,u,3,0)
    assert x.dtype==np.float64 and np.all(x>=0) and np.all(x==0)
    assert row_norms(S).dtype==np.float64


def test_independent_background_means_and_buffer_ownership():
    from src.preprocessing import prepare_system,prepare_measurement
    cal=np.array([[[4+2j,8+1j],[3-1j,7+4j]]],np.complex64);bg=np.array([[[1+0j,3+0j],[0+1j,2+3j]]],np.complex64)
    before=cal.copy();expected=before[0]-np.array([2+0j,1+2j],np.complex64)[:,None]
    np.testing.assert_array_equal(prepare_system(cal,bg),expected);np.testing.assert_array_equal(cal,before)
    result=prepare_system(cal,bg,copy=False);np.testing.assert_array_equal(result,expected);assert np.shares_memory(result,cal)
    fg=np.array([[[2+1j,4+2j],[6+3j,8+4j]]]);empty=np.array([[[1+0j,2+1j],[3+2j,4+3j]]])
    np.testing.assert_array_equal(prepare_measurement(fg,empty,2),[[2+1j,3+1j]][0])
    np.testing.assert_array_equal(prepare_measurement(fg,empty,1),[0+0j,1+0j])
