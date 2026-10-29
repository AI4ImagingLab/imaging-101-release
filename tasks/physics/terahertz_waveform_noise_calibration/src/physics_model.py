"""Explicit Fourier shift, heteroscedastic likelihood and analytic gradient.
Adapted from THzTools contributors (MIT); see LICENSE.
Only the selected full-latent-waveform method is implemented, with no domain imports.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import rfft,irfft,rfftfreq

def shift_waveforms(mu,a,eta,dt):
    """a[j] mu(t-eta[j]); periodic band-limited real FFT convention."""
    x=np.tile(mu,(len(a),1));w=2*np.pi*rfftfreq(len(mu))/dt
    return irfft(rfft(x)*np.exp(-1j*eta[:,None]*w),n=len(mu))*a[:,None]

def derivative(x,dt=1.,order=1):
    n=x.shape[-1];w=2*np.pi*rfftfreq(n)
    return irfft((1j*w)**order*rfft(x),n=n)/dt**order

def variance_components(zeta,noise,dt):
    alpha,beta,tau=noise
    return np.stack([np.full_like(zeta,alpha**2),(beta*zeta)**2,(tau*derivative(zeta,dt))**2])

@dataclass
class Scales:
    logv:np.ndarray
    mu:np.ndarray
    a:np.ndarray
    eta:np.ndarray

def decode(theta,x,scales):
    m,n=x.shape
    variances=np.exp(theta[:3]*scales.logv)
    mu=x[0]-theta[3:3+n]*scales.mu
    a=1.+np.insert(theta[3+n:3+n+m-1]*scales.a,0,0.)
    eta_samples=np.insert(theta[3+n+m-1:]*scales.eta,0,0.)
    return variances,mu,a,eta_samples

def common(theta,x,scales):
    m,n=x.shape;variances,mu,a,eta=decode(theta,x,scales)
    w=2*np.pi*rfftfreq(n);phase=np.exp(1j*np.outer(eta,w));mf=rfft(mu,workers=1)
    zf=((np.conj(phase)*mf).T*a).T
    z=irfft(zf,n=n,workers=1);dz=irfft(1j*w*zf,n=n,workers=1)
    res=x-z;ressq=res**2;v=variances[0]+variances[1]*z**2+variances[2]*dz**2
    return dict(variance_parameters=variances,mu=mu,a=a,eta_samples=eta,zeta=z,dzeta=dz,zeta_f=zf,phase=phase,residual=res,ressq=ressq,variance=v,w=w)

def objective(theta,x,scales):
    c=common(theta,x,scales);v=c['variance'];r2=c['ressq']
    return np.float64(.5*(np.sum(np.log(v))+np.sum(r2/v)))

def gradient(theta,x,scales):
    """Full analytic scaled gradient including signal-dependent noise variance."""
    c=common(theta,x,scales);va,vb,vt=c['variance_parameters'];z=c['zeta'];dz=c['dzeta'];zf=c['zeta_f'];a=c['a'];phase=c['phase'];v=c['variance'];w=c['w'];n=x.shape[1]
    res=c['residual'];reswt=res/v;dvar=(v-c['ressq'])/v**2
    logvar=[.5*np.sum(dvar)*va*scales.logv[0],.5*np.sum(z**2*dvar)*vb*scales.logv[1],.5*np.sum(dz**2*dvar)*vt*scales.logv[2]]
    p=rfft(vb*dvar*z-reswt,workers=1)-1j*vt*w*rfft(dvar*dz,workers=1)
    gmu=-np.sum((irfft(phase*p,n=n,workers=1).T*a).T,axis=0)*scales.mu
    term=np.diag(1/a)@((v-va)*dvar-reswt*z)
    ga=np.sum(term,axis=1)[1:]*scales.a
    ddz=irfft(-w**2*zf,n=n,workers=1)
    geta=-np.sum(dvar*(z*dz*vb+dz*ddz*vt)-reswt*dz,axis=1)[1:]*scales.eta
    return np.concatenate((logvar,gmu,ga,geta))

def initialize(x,dt):
    """Author variance-regression initialization and all optimization scales."""
    m,n=x.shape;mu0=x.mean(axis=0);column=x.T
    v_t=np.var(column,1,ddof=1);sigma_min=np.sqrt(np.min(v_t))
    mu_f=rfft(mu0,workers=1);w=2*np.pi*rfftfreq(n,dt)
    dmu=irfft(1j*w*mu_f,n=n,workers=1)
    A=np.stack([np.ones(n),mu0**2,dmu**2],axis=1)
    sigma_est=np.ma.sqrt(np.linalg.lstsq(A,v_t,rcond=None)[0]).filled(sigma_min)
    alpha,beta,tau=sigma_est
    # Source NoiseModel.noise_amp uses derivative per sample, then divides by dt.
    mu_scale=np.sqrt(alpha**2+(beta*mu0)**2+(tau*derivative(mu0,dt))**2)
    mu_scale[np.isclose(mu_scale,0.)]=np.sqrt(np.finfo(float).eps)
    a_scale=np.full(m-1,np.max((sigma_min/np.max(abs(column)),beta)))
    eta_scale=np.full(m-1,np.max((sigma_min/np.max(abs(np.diff(column,axis=0))),tau/dt)))
    scales=Scales(np.full(3,1/m),mu_scale,a_scale,eta_scale)
    logvar=np.ma.log(np.asarray([alpha,beta,tau/dt])**2).filled(-100.)/scales.logv
    theta=np.r_[logvar,(column[:,0]-mu0)/mu_scale,np.zeros(m-1),np.zeros(m-1)]
    return theta,scales,dict(sigma_est=sigma_est,mu0=mu0,a0=np.ones(m),eta0=np.zeros(m),sigma_min=sigma_min)

def decode_outputs(theta,x,scales,dt):
    m,n=x.shape;v,mu,a,eta_samples=decode(theta,x,scales)
    # The source finite-M correction applies to all three fitted amplitudes.
    noise=np.sqrt(v)*np.array([1.,1.,dt])*np.sqrt(m/(m-1))
    eta=eta_samples*dt;expected=shift_waveforms(mu,a,eta,dt)
    std=np.sqrt(np.sum(variance_components(expected,noise,dt),axis=0))
    return dict(reconstruction=mu[None],amplitude=a[None],delay=eta[None],noise=noise[None],expected_waveforms=expected[None],noise_std=std[None])

def curvature(theta,x,scales,dt,step=1e-3):
    """Five-point derivative of analytic gradient; general local finite differences.

    Curvature is diagnostic only and never replaces the selected BFGS result.
    The source inverse-Hessian coordinate convention uses positive mu scale;
    marginal standard errors are unchanged by that coordinate sign convention.
    """
    H=np.empty((len(theta),len(theta)))
    for j in range(len(theta)):
        e=np.zeros_like(theta);e[j]=step
        H[:,j]=(-gradient(theta+2*e,x,scales)+8*gradient(theta+e,x,scales)-8*gradient(theta-e,x,scales)+gradient(theta-2*e,x,scales))/(12*step)
    hess_inverse=np.linalg.inv(H)
    out=decode_outputs(theta,x,scales,dt);noise=out['noise'][0]
    scale=np.r_[scales.logv*noise/2,scales.mu,scales.a,scales.eta*dt]
    covariance=np.diag(scale)@hess_inverse@np.diag(scale)
    err=np.sqrt(np.diag(covariance));n=x.shape[1];m=x.shape[0]
    return dict(hessian=H,hess_inv=covariance,err_mu=err[3:3+n],err_a=np.r_[0.,err[3+n:3+n+m-1]],err_eta=np.r_[0.,err[3+n+m-1:]],err_noise=err[:3]*np.sqrt(m/(m-1)))
