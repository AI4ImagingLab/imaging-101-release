"""Explicit SHG-FROG mathematics extracted from pypret d47deb6 (MIT).

Copyright (c) 2019 Nils C. Geib.
This research extraction preserves Fourier factors and complex gradients.
"""
import numpy as np
from scipy import fftpack

def abs2(x):
    return x.real*x.real+x.imag*x.imag

def norm2(x):
    return abs2(x).sum()

class FourierGrid:
    def __init__(self,n=512,dw=407474458795.6243):
        self.n=n;self.dw=dw;self.dt=2*np.pi/(n*dw)
        k=np.arange(n);t0=-(n//2)*self.dt;w0=-(n//2)*dw
        self.t=t0+k*self.dt;self.w=w0+k*dw
        self.fr=self.dt*n/(2*np.pi)*np.exp(1j*k*t0*dw)
        self.fs=np.exp(1j*self.t*w0)
        self.br=np.exp(-1j*k*t0*dw)
        self.bs=dw*np.exp(-1j*self.t*w0)
    def forward(self,x):
        return self.fr*fftpack.ifft(self.fs*x)
    def backward(self,x):
        return self.bs*fftpack.fft(self.br*x)

class SHGFrog:
    def __init__(self,grid,delays):
        self.grid=grid;self.delays=np.asarray(delays)
        self.mask=np.exp(1j*self.delays[:,None]*grid.w)
    def signal(self,spectrum,indices=None):
        mask=self.mask if indices is None else self.mask[indices]
        e=self.grid.backward(spectrum);a=self.grid.backward(mask*spectrum)
        return a*e,a,e,mask
    def trace(self,spectrum):
        return abs2(self.grid.forward(self.signal(spectrum)[0]))
    def gradient(self,target,signal,delayed,field,mask):
        d=target-signal;g=self.grid
        return -2*(2*np.pi)*g.dw/g.dt*(mask.conj()*g.forward(d*field.conj())+g.forward(d*delayed.conj()))
    def project(self,measured,signal):
        s=self.grid.forward(signal);a=np.abs(s);f=a>0
        s[~f]=np.sqrt(measured[~f]+0j)
        s[f]=s[f]/a[f]*np.sqrt(measured[f]+0j)
        return self.grid.backward(s)

def scaled_residual(measured,predicted):
    gain=np.sum(measured*predicted)/np.sum(predicted*predicted)
    r=np.sum((measured-gain*predicted)**2)
    return float(np.sqrt(r/(measured.size*measured.max()**2))),float(gain),float(r)
