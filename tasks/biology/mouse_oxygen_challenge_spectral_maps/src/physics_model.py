"""Explicit two-chromophore spectral absorption and source saturation convention.
Extracted from PATATO SpectralUnmixer/SO2Calculator, Thomas Else2023-25, MIT.
"""
import numpy as np
def forward(coefficients,spectra):
    """Return wavelength images(T,W,Z,Y,X) from coefficient fields(T,C,Z,Y,X)."""
    return np.einsum('tc...,cw->tw...',coefficients,spectra)

def oxygenation(coefficients):
    """Source float32 HbO2/(Hb+HbO2), no clipping; zero denominator gives NaN."""
    thb=coefficients[:,0]+coefficients[:,1];thb=thb.copy();thb[thb==0]=np.nan
    return (coefficients[:,1]/thb).astype(np.float32)

def spectral_relative_residual(coefficients,images,spectra,support):
    prediction=forward(coefficients,spectra)
    observed=np.asarray(images,dtype=np.float64)
    return float(np.linalg.norm((prediction-observed)[...,support])/np.linalg.norm(observed[...,support]))

def gas_boundaries(saturation,reference_mask,window=10,sigma=2,skip_start=0):
    """Original reference-curve change-point heuristic, not recorded valve events."""
    from scipy.ndimage import median_filter
    from scipy.signal import fftconvolve
    measurement=np.array(saturation[:,reference_mask],copy=True)
    position=np.arange(window)-window/2+1/2
    kernel=np.exp(-((position/sigma)**2));kernel/=kernel.sum();kernel=kernel[:,None]
    smoothed=median_filter(measurement,kernel.shape)
    smoothed=fftconvolve(smoothed,kernel,'valid')
    gradient=np.median(np.gradient(smoothed,axis=0),axis=-1)
    first=int(np.argmax(gradient));second=first+int(np.argmin(gradient[first:]))
    steps=[skip_start,first+window//2]
    if gradient[second] < -gradient[first]/2 and second-first>first*.5:steps.append(second+window//2)
    steps.append(len(measurement)-1)
    return steps

def window_summary(saturation,steps,buffer_width=5):
    """Return float32 baseline mean/std and challenge-minus-baseline maps, no clipping."""
    baseline=saturation[steps[0]:steps[1]-buffer_width]
    challenge=saturation[steps[1]+buffer_width:steps[2]-buffer_width]
    if len(baseline)==0 or len(challenge)==0:raise ValueError('Empty challenge interval')
    mean=baseline.mean(axis=0)
    return mean,baseline.std(axis=0),challenge.mean(axis=0)-mean

def enhancement_summary(saturation,reference_mask,window=10,sigma=2,buffer_width=5):
    """Return original inferred steps, baseline mean/std, apparent enhancement maps."""
    steps=gas_boundaries(saturation,reference_mask,window,sigma)
    return (steps,*window_summary(saturation,steps,buffer_width))
