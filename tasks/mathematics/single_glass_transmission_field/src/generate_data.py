"""Independent review-only reflection acquisition cases, not bound observations.

The actual benchmark reuses the full real author photograph. Synthetic cases
here expose ambiguity and failure under explicit image-domain assumptions:
finite normalized Gaussian PSF, reflected boundaries, alpha-weighted focused
transmission plus blurred reflection, optional zero-mean Gaussian noise in
normalized detector units. No clipping is applied to these scientific arrays.
Known layers are retained only for hidden maintainer evaluation. They are
not measurements, real-scene truth, or an instrument-calibrated noise model.
"""
import numpy as np
from scipy.ndimage import convolve
def perturb_observation(observation,sigma,seed=2019):
    """Illustrative normalized-detector Gaussian stress, without output clipping."""
    if not np.isfinite(sigma) or sigma<0:raise ValueError('sigma must be finite and nonnegative')
    return np.asarray(observation,dtype=float)+np.random.default_rng(seed).normal(0,sigma,np.shape(observation))
def gaussian_blur(image,sigma):
    if sigma<=0 or not np.isfinite(sigma):raise ValueError('sigma must be positive and finite')
    radius=int(np.ceil(4*sigma));axis=np.arange(-radius,radius+1,dtype=float);one=np.exp(-axis*axis/(2*sigma*sigma));one/=one.sum()
    return convolve(convolve(np.asarray(image,dtype=float),one[:,None],mode='reflect'),one[None,:],mode='reflect')
def synthetic_mixture(shape=(96,128),alpha=.75,sigma=5.,noise=0.,weak=False,seed=2019):
    """Return complete known layers/mixture; no hidden truth used by inverse."""
    if not 0<alpha<1 or noise<0 or not np.isfinite(noise):raise ValueError('Invalid mixture parameters')
    yy,xx=np.indices(shape,dtype=float);yy/=shape[0];xx/=shape[1]
    contrast=.035 if weak else .5
    transmission=.18+contrast*((xx>.2)&(xx<.72)&(yy>.18)&(yy<.82))+.04*np.sin(12*np.pi*xx)
    reflection=.2+.7*(((xx-.7)**2+(yy-.3)**2)<.12**2)+.2*np.sin(4*np.pi*yy)*np.cos(3*np.pi*xx)
    blurred=gaussian_blur(reflection,sigma);rng=np.random.default_rng(seed)
    observation=alpha*transmission+(1-alpha)*blurred+rng.normal(0,noise,shape)
    return {'observation':observation,'transmission':transmission,'reflection':reflection,'blurred_reflection':blurred,'transmitted_component':alpha*transmission,'mean_convention_target':alpha*transmission+(1-alpha)*blurred.mean(),'alpha':alpha,'sigma':sigma}
