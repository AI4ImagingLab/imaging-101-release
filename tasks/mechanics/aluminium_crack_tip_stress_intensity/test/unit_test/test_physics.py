import numpy as np
import pytest
from src.physics_model import displacement_basis,stress_field,stress_intensity,TERMS,YOUNG_MODULUS,POISSON_RATIO
from src.preprocessing import transform,fitting_grid

def test_hooke_law_independent_cartesian_finite_difference():
    x=np.array([.9,1.3,2.1]);y=np.array([.4,-.6,.3]);h=1e-5
    c=np.array([2.,31.,-4.,.3,.2,-.1,3.,-12.,2.,.1,.03,.05])
    def u(x,y):
        bx,by=displacement_basis(np.hypot(x,y),np.arctan2(y,x));return np.array([c@bx,c@by])
    dx=(u(x+h,y)-u(x-h,y))/(2*h);dy=(u(x,y+h)-u(x,y-h))/(2*h)
    E=YOUNG_MODULUS;nu=POISSON_RATIO
    hooke=E/(1-nu*nu)*np.array([dx[0]+nu*dy[1],nu*dx[0]+dy[1],(1-nu)/2*(dx[1]+dy[0])])
    np.testing.assert_allclose(stress_field(c,np.hypot(x,y),np.arctan2(y,x)),hooke,rtol=2e-8,atol=1e-7)

@pytest.mark.parametrize('i',range(12))
def test_traction_free_crack_faces(i):
    c=np.zeros(12);c[i]=1
    field=stress_field(c,np.ones(2)*1.3,np.array([-np.pi,np.pi]))
    np.testing.assert_allclose(field[1:],0,atol=1e-13)

def test_singular_mode_units_sign_and_axis():
    c=np.zeros(12);c[1]=2;c[7]=3
    field=stress_field(c,np.array([4.]),np.array([0.]))
    np.testing.assert_allclose(field[:,0],[1,1,-1.5])
    np.testing.assert_allclose(stress_intensity(c),np.sqrt(2*np.pi/1000)*np.array([2,-3]))

def test_rigid_rotation_and_dimensions():
    p=np.array([[15.39,.4],[16.39,.4]]);u=np.array([[1.,2.],[3.,4.]])
    x,v=transform(p,u);np.testing.assert_allclose(x[0],0)
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),np.linalg.norm(u,axis=1))
    r,phi,x,y=fitting_grid();assert r.shape==(231,559)
    np.testing.assert_allclose(np.hypot(x,y),r)

def test_linear_operator_adjoint_and_loss_gradient():
    r=np.array([1.,2.,3.]);phi=np.array([-.8,.2,.9]);bx,by=displacement_basis(r,phi)
    A=np.concatenate([bx,by],axis=1).T;c=np.linspace(-1,2,12);v=np.arange(6.)
    assert np.dot(A@c,v)==pytest.approx(np.dot(c,A.T@v),abs=1e-16)
    b=np.arange(6.)/100;d=np.linspace(.1,.2,12);h=1e-5
    loss=lambda q:np.sum((A@q-b)**2)/2
    assert (loss(c+h*d)-loss(c-h*d))/(2*h)==pytest.approx(d@(A.T@(A@c-b)),rel=1e-6,abs=1e-12)
