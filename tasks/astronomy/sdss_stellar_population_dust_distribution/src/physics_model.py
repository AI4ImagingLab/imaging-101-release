"""Explicit FIREFLY spectral operators; MIT FIREFLY Collaboration 2026."""
import numpy as np
def highpass(flux,windowsize=20,start=20):
    n=len(flux);window=np.zeros(n);window[0]=1
    for i in range(windowsize):
        window[start+i]=(i+1.)/windowsize
        window[n-1-(start+windowsize-i)]=(windowsize-i)/windowsize
    window[start+windowsize:n-start-windowsize]=1
    complement=1-window;complement[0]=1
    transformed=np.fft.fft(flux);hi=np.fft.ifft(transformed*window).real;lo=np.fft.ifft(transformed*complement).real
    result=(1+(hi-np.median(hi))/lo)*np.median(hi)
    if not np.isfinite(result).all():raise ValueError('Nonfinite highpass data')
    return result
def calzetti(ebv,wavelength):
    l=wavelength/10000.;k=np.zeros_like(l);red=(l>=.63)&(l<=2.2);blue=l<.63
    k[red]=2.659*(-1.857+1.040/l[red])+4.05
    k[blue]=2.659*(-2.156+1.509/l[blue]-.198/l[blue]**2+.011/l[blue]**3)+4.05
    return 10**(-.4*ebv*k)
def smooth_curve(wavelength,curve,length=200.):
    return np.array([np.median(curve[(wavelength<x+length)&(wavelength>x-length)]) for x in wavelength])
def forward(weights,attenuation,templates,flux):
    models=templates*attenuation
    models=models/np.median(models,axis=1)[:,None]*np.median(flux)
    return weights@models
def adjoint(residual,normalized_templates):return normalized_templates@residual
