import numpy as np
from src.physics_model import shift_waveforms,derivative,variance_components,initialize,gradient,objective,decode_outputs

def test_periodic_shift_derivative_and_adjoint():
    n=17;dt=.1;t=np.arange(n)*dt;w=2*np.pi*2/(n*dt);mu=np.cos(w*t)
    np.testing.assert_allclose(derivative(mu,dt),-w*np.sin(w*t),atol=1e-12)
    delay=np.array([0.,3*dt,.017]);a=np.array([1.,.7,1.3]);out=shift_waveforms(mu,a,delay,dt)
    np.testing.assert_allclose(out[1],.7*np.roll(mu,3),atol=1e-12)
    np.testing.assert_allclose(out[2],1.3*np.cos(w*(t-delay[2])),atol=1e-12)
    rng=np.random.default_rng(73);u=rng.normal(size=n);v=rng.normal(size=n);eta=.071;scale=1.2
    Au=shift_waveforms(u,np.array([scale]),np.array([eta]),dt)[0];ATv=shift_waveforms(v,np.array([scale]),np.array([-eta]),dt)[0]
    np.testing.assert_allclose(Au@v,u@ATv,atol=1e-12)
    np.testing.assert_allclose(derivative(u,dt)@v,-u@derivative(v,dt),atol=1e-12)

def test_noise_components_bias_and_full_gradient():
    rng=np.random.default_rng(23);n=15;m=5;dt=.08;t=np.arange(n)*dt;mu=2*np.sin(2*np.pi*t/(n*dt))
    x=np.tile(mu,(m,1))+.2*rng.standard_normal((m,n))
    theta,s,_=initialize(x,dt);g=gradient(theta,x,s)
    for _ in range(5):
        v=rng.normal(size=theta.size);v/=np.linalg.norm(v);h=1e-4
        numeric=(objective(theta+h*v,x,s)-objective(theta-h*v,x,s))/(2*h)
        np.testing.assert_allclose(g@v,numeric,rtol=1e-6,atol=1e-7)
    output=decode_outputs(theta,x,s,dt)
    np.testing.assert_array_equal(output['amplitude'][0,0],1.);np.testing.assert_array_equal(output['delay'][0,0],0.)
    expected=np.sqrt(np.exp(theta[:3]*s.logv))*np.array([1,1,dt])*np.sqrt(m/(m-1))
    np.testing.assert_allclose(output['noise'][0],expected)
    noise=np.array([.2,.03,.002]);parts=variance_components(np.ones((2,15))*3,noise,dt)
    np.testing.assert_allclose(parts[0],.04);np.testing.assert_allclose(parts[1],.0081);np.testing.assert_allclose(parts[2],0,atol=1e-30)
