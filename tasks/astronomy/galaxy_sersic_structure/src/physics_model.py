"""Explicit pixel-integrated Sersic brightness and calibrated PSF convolution.
Adapted from PySersic, MIT LICENSE; PixelRenderer mathematical path only.
"""
import numpy as np
import torch
def sersic(x,y,parameters):
    """Surface brightness in nanomaggies/pixel^2; first7 physical parameters."""
    xc,yc,flux,re,n,ellip,theta=parameters[:7]
    # Published implementation's high-accuracy analytic b_n approximation.
    bn=2*n-1/3+4/(405*n)+46/(25515*n**2)+((.00018073182821237496/(n+torch.exp(n))+3.7026973571904255e-5)/(-.09149183119702775-n)**2)*((torch.log(n)+(n*2.6248718397705195-.9727511612512357)/(n**(-4)+94.78011643586419))/(n-.006044236674273689))
    angle=theta+torch.pi/2;c=torch.cos(angle);s=torch.sin(angle)
    major=(x-xc)*c+(y-yc)*s;minor=-(x-xc)*s+(y-yc)*c
    radius=torch.sqrt((major/re)**2+(minor/((1-ellip)*re))**2)
    amplitude=flux*bn**(2*n)/(torch.exp(bn+torch.lgamma(2*n))*re**2*torch.pi*2*n)
    return amplitude*torch.exp(-bn*(radius**(1/n)-1))/(1-ellip)
class PixelModel:
    def __init__(self,psf,shape=(120,120),num_os=12,os_pixel_size=6,dtype=torch.float32):
        """Fixed12x12 central pixels integrated using12x12 Gauss-Legendre."""
        self.shape=shape;self.dtype=dtype;self.os=os_pixel_size
        y,x=torch.meshgrid(torch.arange(shape[0],dtype=dtype),torch.arange(shape[1],dtype=dtype),indexing='ij');self.x=x;self.y=y
        nodes,weights=np.polynomial.legendre.leggauss(num_os);nodes=torch.tensor(nodes,dtype=dtype)/2;weights=torch.tensor(weights,dtype=dtype)/2
        dy,dx=torch.meshgrid(nodes,nodes,indexing='ij');wy,wx=torch.meshgrid(weights,weights,indexing='ij');self.weights=wx*wy
        cy,cx=shape[0]//2,shape[1]//2;self.region=(slice(cy-os_pixel_size,cy+os_pixel_size),slice(cx-os_pixel_size,cx+os_pixel_size))
        self.xos=x[self.region][...,None,None]+dx;self.yos=y[self.region][...,None,None]+dy
        fy,fx=torch.meshgrid(torch.fft.fftfreq(shape[0],dtype=dtype),torch.fft.rfftfreq(shape[1],dtype=dtype),indexing='ij')
        p=torch.as_tensor(psf,dtype=dtype);shiftx=torch.exp(1j*2*3.1415*(p.shape[1]/2-.5)*fx);shifty=torch.exp(1j*2*3.1415*(p.shape[0]/2-.5)*fy)
        # Preserve original3.1415 centering constant for numerical parity.
        # It implies a known0.000914px shift peraxis for63pxPSF, quantified
        # independently in validation; this is not silently treated as exactpi.
        self.psf_fft=torch.fft.rfft2(p,s=shape)*shiftx*shifty
    def intrinsic(self,parameters):
        im=sersic(self.x,self.y,parameters);central=(sersic(self.xos,self.yos,parameters)*self.weights).sum(dim=(-1,-2))
        result=im.clone();result[self.region]=central;return result
    def observed(self,parameters):
        intrinsic=self.intrinsic(parameters)
        return torch.fft.irfft2(torch.fft.rfft2(intrinsic)*self.psf_fft,s=self.shape)+parameters[7]
def objective(parameters,model,image,rms,valid):
    predicted=model.observed(parameters);residual=(predicted-image)/rms
    likelihood=(.5*residual**2+torch.log(rms)+.5*np.log(2*np.pi))[valid].sum()
    prior=.5*(((parameters[:2]-60)/5)**2).sum()+.5*(parameters[7]/.01)**2
    return likelihood+prior
