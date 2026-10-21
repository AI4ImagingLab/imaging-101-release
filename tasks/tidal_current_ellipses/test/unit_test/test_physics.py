import numpy as np
from src.preprocessing import polar_to_velocity
from src.physics_model import harmonic_basis,ellipse_parameters,rotary_from_coefficients,coefficients_from_rotary,predict
from src.solvers import solve
def test_display_metrics_known_independent_case():
    from src.visualization import metrics
    assert metrics([1,0],[0,1])==dict(ncc=0.,nrmse=1.)
def test_polar_directions_and_rotary_invertibility():
    np.testing.assert_allclose(polar_to_velocity(np.ones(4),np.array([0,90,180,270])),[[0,1,0,-1],[1,0,-1,0]],atol=1e-14)
    c=np.array([[1,2],[3,4.]]);s=np.array([[-1,2],[2,-1.]]);mean=np.array([3.,4.]);m=rotary_from_coefficients(c,s,mean)
    for x,y in zip(coefficients_from_rotary(m),(c,s,mean)):np.testing.assert_allclose(x,y,atol=1e-14)
def test_analytic_ellipse_orientation_rotation_and_degeneracy():
    theta=np.pi/4
    for minor in (1.,-1.):
        c=np.array([[3*np.cos(theta)],[3*np.sin(theta)]]);s=np.array([[-minor*np.sin(theta)],[minor*np.cos(theta)]])
        np.testing.assert_allclose(ellipse_parameters(c,s),[[3,minor,45,0]],atol=1e-12)
    np.testing.assert_array_equal(ellipse_parameters(np.zeros((2,1)),np.zeros((2,1))),np.zeros((1,4)))
    e=ellipse_parameters(np.array([[2.],[0.]]),np.array([[0.],[2.]]));np.testing.assert_allclose(e[0,:2],[2,2])
def test_gappy_analytic_inverse_normal_equations_and_adjoint():
    rng=np.random.default_rng(492);t=np.sort(rng.uniform(0,90*86400,500));f=np.array([.08,.04]);b,tref=harmonic_basis(t,f)
    m=np.array([1+2j,3-1j,2-.5j,-1+2j,.2+.7j]);z=b@m
    r,d=solve(t,np.array([z.real,z.imag]),f,return_diagnostics=True)
    np.testing.assert_allclose(d['m'],m,rtol=1e-12,atol=1e-12)
    assert np.linalg.norm(b.conj().T@d['residual'])<1e-10
    x=rng.normal(size=5)+1j*rng.normal(size=5);y=rng.normal(size=500)+1j*rng.normal(size=500)
    np.testing.assert_allclose(np.vdot(b@x,y),np.vdot(x,b.conj().T@y),rtol=1e-12)
    np.testing.assert_allclose(predict(t,f,r['cosine'][0],r['sine'][0],r['mean_current'][0],tref),r['reconstruction'][0],atol=1e-12)
