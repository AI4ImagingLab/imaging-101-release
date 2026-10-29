from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import spectrum,allan,FS,KT
from src.physics_model import prediction,negative_log_likelihood,derivatives
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_all_original_stages():
    u=np.load(ROOT/'test/fixtures/upstream.npz')
    for kind,prepare in [('psd',spectrum),('av',allan)]:
        d=prepare(u['trace'])
        for noise in [False,True]:
            p=kind+('_noise' if noise else '_zero')
            for k in ('x','y','shape'):assert_allclose(d[k],u[p+'_'+k],rtol=1e-8,atol=1e-10)
            assert_allclose(prediction(d['x'],u[p+'_guess'],kind),u[p+'_initial_model'],rtol=1e-8,atol=1e-10)
            assert_allclose(negative_log_likelihood(u[p+'_guess'],d,kind),u[p+'_initial_nll'],rtol=1e-8,atol=1e-10)
            # Isolate differentiation on the independently frozen original statistics.
            original_data={k:u[p+'_'+k] for k in ('x','y','shape')}
            gradient,hessian=derivatives(u[p+'_params'],original_data,kind)
            assert_allclose(gradient,u[p+'_gradient'],rtol=1e-8,atol=1e-10)
            assert_allclose(hessian,u[p+'_hessian'],rtol=1e-8,atol=1e-10)
            result=reconstruct(u['trace'],kind,noise)
            assert_allclose(result['parameters'],u[p+'_params'],rtol=1e-4,atol=0)
def test_exposure_covariance_independent_spectrum():
    p=np.array([1.8e-5,1.8e-4,8.]);g,k,e=p;r=k/(g*FS);variance=KT/k
    r0=2*variance*(r+np.expm1(-r))/r**2
    lags=np.arange(1,3001);cov=variance*np.exp(-lags*r)*(np.exp(r)+np.exp(-r)-2)/r**2
    f=np.array([.5,5.,50.,150.])
    expected=(r0+2*np.cos(2*np.pi*f[:,None]*lags/FS)@cov+e**2)/FS
    assert_allclose(prediction(f,p),expected,rtol=1e-8,atol=1e-10)
def test_likelihood_gradient_finite_difference():
    u=np.load(ROOT/'test/fixtures/upstream.npz');d=spectrum(u['trace']);p=u['psd_noise_params']*np.array([1.1,.9,1.2]);g,h=derivatives(p,d)
    for j in range(3):
        delta=np.zeros(3);delta[j]=p[j]*1e-5
        finite=(negative_log_likelihood(p+delta,d)-negative_log_likelihood(p-delta,d))/(2*delta[j])
        assert_allclose(finite,g[j],rtol=2e-4,atol=1e-7)
def test_allan_direct_blocks_and_parseval():
    u=np.load(ROOT/'test/fixtures/upstream.npz');x=u['trace'];a=allan(x)
    for j,m in enumerate([1,2,4]):
        means=np.convolve(x,np.ones(m)/m,'valid')
        v=np.mean((means[m:]-means[:-m])**2)/2
        assert_allclose(a['y'][j],v,rtol=1e-10)
    n=int(2*len(x)/36);window=.5-.5*np.cos(2*np.pi*np.arange(n)/n);w=x[:n]-x[:n].mean()
    full=np.abs(np.fft.fft(w*window))**2/(FS*np.sum(window**2))
    assert_allclose(full.sum()*FS/n,np.sum((w*window)**2)/np.sum(window**2),rtol=1e-14)
