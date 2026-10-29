import numpy as np
import pytest
from src.preprocessing import modified_laplacian, initial_focal_coordinate, polynomial_operator
from src.physics_model import gradient, adjoint, solve_neumann, polynomial_value
from src.solvers import shrink,tv_denoise_only,classical_focus

def test_modified_laplacian_known_rgb():
    a=np.zeros((1,5,6,3),np.uint8);a[0,2,3]=255
    s=modified_laplacian(a)
    assert s[0,2,3]==12
    assert s[0,2,2]==3 and s[0,1,3]==3
    assert s[0,0,0]==0

def test_replicate_boundary():
    a=np.zeros((1,3,3,3),np.uint8);a[0,0,0,0]=255
    s=modified_laplacian(a)
    assert s[0,0,0]==2 and s[0,1,0]==1 and s[0,0,1]==1

def test_gradient_adjoint_including_arbitrary_boundary():
    r=np.random.default_rng(420);u=r.normal(size=(7,9));a=r.normal(size=u.shape);b=r.normal(size=u.shape)
    x,y=gradient(u)
    np.testing.assert_allclose(np.sum(x*a+y*b),np.sum(u*adjoint(a,b)),rtol=1e-13,atol=1e-13)

def test_dense_neumann_inverse():
    shape=(4,5);n=np.prod(shape);eye=np.eye(n)
    L=np.column_stack([adjoint(*gradient(e.reshape(shape))).ravel() for e in eye])
    rhs=np.arange(n,dtype=float).reshape(shape)/7
    np.testing.assert_allclose(solve_neumann(rhs,2.3),np.linalg.solve(eye+2.3*L,rhs.ravel()).reshape(shape),rtol=2e-13,atol=2e-13)

def test_constant_neumann_nullspace():
    np.testing.assert_allclose(solve_neumann(np.full((7,9),2.1),12.7),2.1,atol=1e-13)

def test_polynomial_derivative_independent_difference():
    r=np.random.default_rng(49);c=r.normal(size=(5,3,4));x=r.uniform(-.8,.8,size=(3,4));h=1e-6
    d=c[:-1]*np.arange(4,0,-1)[:,None,None]
    np.testing.assert_allclose((polynomial_value(c,x+h)-polynomial_value(c,x-h))/(2*h),polynomial_value(d,x),rtol=1e-8,atol=1e-8)

@pytest.mark.parametrize('a,b,t',[(3.,4.,2.),(0.,0.,1.),(.1,.2,1.),(-3.,4.,2.)])
def test_isotropic_proximal_exact(a,b,t):
    x,y=shrink(np.array([[a]]),np.array([[b]]),t)
    factor=max(0,1-t/max(np.hypot(a,b),1e-300))
    np.testing.assert_allclose([x.item(),y.item()],[factor*a,factor*b],atol=1e-15)

def test_explicit_original_initializer_offset():
    a=np.zeros((19,9,11),np.float32);a[7]=1
    u=initial_focal_coordinate(a)
    np.testing.assert_allclose(u,20*((7-1)/18-.5),atol=2e-5)
    corrected=initial_focal_coordinate(a,index_offset=0)
    np.testing.assert_allclose(corrected-u,20/18,atol=2e-5)

def test_polynomial_constant_and_quadratic():
    X,P=polynomial_operator(19,8)
    node=X[:,-2].astype(float)
    predicted=X.astype(float)@(P.astype(float)@(1-.003*(node-2)**2))
    np.testing.assert_allclose(predicted,1-.003*(node-2)**2,atol=.002,rtol=0)


def test_rof_two_pixel_analytic_optimum():
    result=tv_denoise_only(np.array([[-1.,1.]],np.float32),.4,250)
    np.testing.assert_allclose(result,[[-.6,.6]],atol=2e-6,rtol=0)


def test_classical_unshifted_focus_coordinates():
    data=np.zeros((19,5,7),np.float32);data[7]=1
    result=classical_focus(data,3,3)
    np.testing.assert_allclose(result,20*(7/18-.5),atol=2e-6,rtol=0)


def test_figure_metrics_independent_known_vector():
    from src.visualization import image_metrics
    a=np.array([0.,1.,2.]);b=np.array([0.,2.,2.])
    m=image_metrics(a,b)
    assert m['ncc']==pytest.approx(6/np.sqrt(40))
    assert m['nrmse']==pytest.approx(np.sqrt(1/3)/2)


def test_approximate_fma_double_rounding_limit_is_explicit():
    from fractions import Fraction
    from src.preprocessing import fused_multiply_add
    a=np.float32((1+2**-23)*2**-12)
    b=np.float32((1-2**-23)*2**-12)
    c=np.float32(1+2**-23)
    exact=Fraction(float(a))*Fraction(float(b))+Fraction(float(c))
    lower=np.float32(1+2**-23);upper=np.float32(1+2*2**-23)
    midpoint=(Fraction(float(lower))+Fraction(float(upper)))/2
    assert exact<midpoint and midpoint-exact==Fraction(1,2**70)
    assert fused_multiply_add(a,b,c)==upper
    assert lower!=upper  # A true single-rounded FMA would choose lower.
