"""Analytic physical checks, independent of upstream numerical agreement."""
import numpy as np
from src.physics_model import nearest_integer,beam_kernel,beam_convolve,CloudRandom
from src.solvers import projected_velocity,prepass_derivatives


def test_halfway_rounding_is_away_from_zero():
    np.testing.assert_array_equal(nearest_integer([-1.5,-.5,.5,1.5]),[-2,-1,1,2])


def test_normalized_positive_beam_and_zero_boundary():
    kernel=beam_kernel()
    assert abs(kernel.sum()-1)<1e-14 and (kernel>=0).all()
    np.testing.assert_allclose(kernel,kernel[::-1,::-1],rtol=0,atol=1e-15)
    impulse=np.zeros((1,31,31),dtype=np.float32);impulse[0,15,15]=1
    result=beam_convolve(impulse,kernel)
    assert abs(float(result.sum())-1)<1e-6
    assert result[0,0,0]==0 and result[0,-1,-1]==0


def test_major_axis_doppler_sign_and_minor_axis_projection():
    p=np.array([100,40,0,90,60,0,0],dtype=np.float32)
    points=np.array([[-1,0],[1,0],[0,1]],dtype=np.float32)
    expected=np.array([100+40*np.sqrt(3)/2,100-40*np.sqrt(3)/2,100])
    np.testing.assert_allclose(projected_velocity(points,p),expected,atol=1e-5,rtol=0)


def test_analytic_velocity_derivatives():
    p=np.array([132.8,120,0,123.7,60,77,77],dtype=np.float32)
    coordinates=np.array([[70,72],[80,90],[60,87]],dtype=np.float32)
    analytic=prepass_derivatives(coordinates,p)
    for j,index in enumerate([1,3]):
        plus=p.copy();minus=p.copy();plus[index]+=.05;minus[index]-=.05
        numerical=(projected_velocity(coordinates,plus).astype(float)-projected_velocity(coordinates,minus).astype(float))/float(plus[index]-minus[index])
        np.testing.assert_allclose(analytic[:,j],numerical,rtol=0,atol=4e-4)


def test_general_random_quadrature_distributions():
    # Distributional checks, not seed equality or a seed-based quality claim.
    random=CloudRandom(103)
    u=random.uniform(100000).astype(float)
    assert abs(u.mean())<.008 and abs(u.var()-1/3)<.008
    z=np.arctanh(u)
    assert abs(z.mean())<.015 and abs(z.var()-np.pi**2/12)<.025
    normal=random.normal(100000).astype(float)
    assert abs(normal.mean())<.012 and abs(normal.var()-1)<.025


def test_gauss_jordan_known_damped_system():
    from src.solvers import _gauss_jordan_inverse
    a=np.array([[2.,.3],[.3,4.]])
    np.testing.assert_allclose(_gauss_jordan_inverse(a)@a,np.eye(2),rtol=0,atol=1e-14)
