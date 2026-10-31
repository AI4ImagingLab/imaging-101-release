import numpy as np
import pytest
from src.physics_model import KERNELS,covariance_operators,acceleration_parameter,q_delta_standard_deviation
from src.solvers import negative_log_likelihood
@pytest.mark.parametrize('kernel',KERNELS)
def test_independent_derivatives(kernel):
    x=np.array([.1,.5,1.2]);y=np.array([.2,.9]);theta=np.array([110.,.8]);step=1e-5
    k,g,d,m=covariance_operators(x,y,theta,kernel)
    for j in range(2):
        delta=np.eye(2)[j]*step
        finite=(covariance_operators(x,y,theta+delta,kernel)[0]-covariance_operators(x,y,theta-delta,kernel)[0])/(2*step)
        np.testing.assert_allclose(g[j],finite,rtol=1e-5,atol=1e-5)
    finite=(covariance_operators(x,y+step,theta,kernel)[0]-covariance_operators(x,y-step,theta,kernel)[0])/(2*step)
    np.testing.assert_allclose(d,finite,rtol=1e-5,atol=1e-5)
    finite=(covariance_operators(x+step,y,theta,kernel)[2]-covariance_operators(x-step,y,theta,kernel)[2])/(2*step)
    np.testing.assert_allclose(m,finite,rtol=1e-5,atol=1e-4)
    z=np.linspace(.1,1.5,8);yy=70*(1+z)**1.5;ee=np.full(8,8.)
    _,analytic=negative_log_likelihood(theta,z,yy,ee,kernel)
    numeric=[]
    for j in range(2):
        delta=np.eye(2)[j]*step
        numeric.append((negative_log_likelihood(theta+delta,z,yy,ee,kernel)[0]-negative_log_likelihood(theta-delta,z,yy,ee,kernel)[0])/(2*step))
    np.testing.assert_allclose(analytic,numeric,rtol=1e-5,atol=1e-6)
def test_independent_expansion_laws_and_delta_variance():
    z=np.linspace(0,2,50);h=70*(1+z)**1.5;hp=105*np.sqrt(1+z)
    np.testing.assert_allclose(acceleration_parameter(z,h,hp),.5,atol=1e-15)
    np.testing.assert_allclose(acceleration_parameter(z,np.full(50,70.),np.zeros(50)),-1)
    omega=.3;h=70*np.sqrt(omega*(1+z)**3+1-omega);hp=1.5*70**2*omega*(1+z)**2/h
    np.testing.assert_allclose(acceleration_parameter(z,h,hp),1.5*omega*(1+z)**3/(omega*(1+z)**3+1-omega)-1)
    covariance=np.array([[4.,-.5],[-.5,9.]])
    jac=np.array([-hp[0]/h[0]**2,1/h[0]])
    np.testing.assert_allclose(q_delta_standard_deviation(z[0],h[0],hp[0],covariance)**2,jac@covariance@jac)
