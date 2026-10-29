"""Local Fourier carrier acquisition and periodic gradient integration.

Adapted from the MIT Trapped-Modes-LTG reference. Axis order is always y,x.
"""
import numpy as np
from scipy.fft import fft2,ifft2,fftfreq,fftshift,ifftshift
from scipy.ndimage import label
from .preprocessing import SQUARE_SIZE,LAYERS
from .unwrapping import unwrap_reliability

def frequencies(shape,spacing=1.,shifted=False):
    axes=[fftfreq(n,spacing/(2*np.pi)) for n in shape]
    if shifted:axes=[fftshift(a) for a in axes]
    return np.meshgrid(*axes,indexing='ij')
def find_peaks(image):
    spectrum=fftshift(abs(fft2(image-np.mean(image))));ky,kx=frequencies(image.shape,shifted=True)
    spectrum*=ky*ky+kx*kx>(4*np.pi/min(image.shape))**2
    mask=spectrum>.5*np.max(spectrum);mask[0]=False;mask[-1]=False;mask[:,0]=False;mask[:,-1]=False
    blobs,count=label(mask,np.ones((3,3)));peaks=[]
    for i in range(1,count+1):
        coordinates=np.argwhere(blobs==i);p=coordinates[np.argmax(spectrum[blobs==i])];peaks.append((spectrum[tuple(p)],p))
    locations=[p for _,p in sorted(peaks,key=lambda x:x[0])[:4]]
    ky,kx=frequencies(image.shape,shifted=True)
    vector=lambda p:np.array([ky[tuple(p)],kx[tuple(p)]])
    first=min(locations,key=lambda p:abs(np.arctan2(*vector(p))))
    second=min(locations,key=lambda p:abs(np.dot(vector(first),vector(p))))
    return np.array([first,second])
def carriers(reference,square_size=SQUARE_SIZE,radius_factor=1.):
    peaks=find_peaks(reference);ky,kx=frequencies(reference.shape,shifted=True);pixel=np.array([[ky[tuple(p)],kx[tuple(p)]] for p in peaks])
    wavelength=2*np.pi/np.mean(abs(pixel));calibration=2*square_size/wavelength
    radius=np.linalg.norm(peaks[0]-peaks[1])/2*radius_factor;rows,cols=np.indices(reference.shape)
    masks=np.array([ifftshift((rows-p[0])**2+(cols-p[1])**2<radius*radius) for p in peaks]);ft=fft2(reference)
    return dict(peaks=peaks,calibration=calibration,frequencies=pixel/calibration,masks=masks,reference_carriers=np.array([np.conj(ifft2(ft*m)) for m in masks]))
def phases(displaced,carrier,unwrap=True):
    ft=fft2(displaced);result=[]
    for mask,reference in zip(carrier['masks'],carrier['reference_carriers']):
        phase=-np.angle(ifft2(ft*mask)*reference)
        result.append(unwrap_reliability(phase) if unwrap else phase)
    return np.asarray(result,dtype='float64')
def displacement(phase,carrier):
    q=carrier['frequencies'];det=q[0,1]*q[1,0]-q[0,0]*q[1,1]
    return np.array([(q[1,0]*phase[0]-q[0,0]*phase[1])/det,(q[0,1]*phase[1]-q[1,1]*phase[0])/det])
def effective_height(layers=LAYERS):
    alpha=1-layers[-1][1]/layers[-2][1]
    return alpha*sum(layers[2][1]*thickness/index for thickness,index in layers[:-1])
def integrate_gradient(gx,gy,spacing):
    if any(n%2==0 for n in gx.shape):raise ValueError('validated integrator requires odd grid dimensions')
    ky,kx=frequencies(gx.shape,spacing);k2=kx*kx+ky*ky;k2[0,0]=1
    return ifft2((-1j*kx*fft2(gx)-1j*ky*fft2(gy))/k2).real

def height_gradient(height,spacing):
    ky,kx=frequencies(height.shape,spacing);ft=fft2(height)
    return np.array([ifft2(1j*kx*ft).real,ifft2(1j*ky*ft).real])
