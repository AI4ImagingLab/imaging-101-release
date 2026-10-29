"""QDM preprocessing and scale-space anomaly detection.

Method adapted from Souza-Junior and Uieda (2023), MIT.
Scale-space peak/pruning conventions follow scikit-image, BSD-3-Clause.
"""
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.spatial import cKDTree

SPACING = 2.35
HEIGHT = 5.0
CONTINUATION = 5.0

def load_data(path='data/raw_data.npz'):
    """Return float64 Bz (nT), x/y (um); reject malformed acquisition."""
    with np.load(path, allow_pickle=False) as d:
        bz, x, y = (d[k][0] for k in ('magnetic_field','x','y'))
    if bz.shape != (len(y),len(x)) or not np.isfinite(bz).all():
        raise ValueError('Invalid magnetic field')
    return bz,x,y

def upward(field, spacing=SPACING, displacement=CONTINUATION):
    """Periodic unpadded Fourier continuation; nT -> nT."""
    ky=2*np.pi*np.fft.fftfreq(field.shape[0],spacing)
    kx=2*np.pi*np.fft.fftfreq(field.shape[1],spacing)
    return np.fft.ifft2(np.fft.fft2(field)*np.exp(-displacement*np.hypot(ky[:,None],kx))).real

def prepare(field,x,y):
    """Return continued Bz, Cartesian gradients (nT/um), TGA stretch."""
    s=float(np.mean([abs(x[1]-x[0]),abs(y[1]-y[0])]))
    f=upward(field,s,CONTINUATION)
    dy,dx=np.gradient(f,y,x)
    dz=(upward(f,s,s)-upward(f,s,-s))/(2*s)
    tga=np.sqrt(dx*dx+dy*dy+dz*dz)
    lo,hi=np.percentile(tga,(1,99))
    stretched=(np.clip(tga,lo,hi)-lo)/(hi-lo)
    return f,dx,dy,dz,tga,stretched

def detect_windows(stretched,x,y):
    """Return intensity-ordered (xmin,xmax,ymin,ymax) windows in um.

    Ten linear Gaussian scales 10..100 um; LoG threshold .025, zero
    fractional overlap, 30 um image-border exclusion, window factor 1.3.
    """
    s=float(np.mean([abs(x[1]-x[0]),abs(y[1]-y[0])]))
    sigmas=np.linspace(10/s,100/s,10)
    cube=np.stack([-gaussian_laplace(stretched,v)*v*v for v in sigmas],axis=-1)
    mask=(cube==maximum_filter(cube,size=3,mode='nearest')) & (cube>.025)
    border=int(30/s)
    mask[:border]=False;mask[-border:]=False
    mask[:,:border]=False;mask[:,-border:]=False
    peaks=np.column_stack(np.nonzero(mask))
    values=cube[tuple(peaks.T)]
    peaks=peaks[np.argsort(-values,kind='stable')]
    blobs=np.column_stack((peaks[:,:2],sigmas[peaks[:,2]]))
    pairs=list(cKDTree(blobs[:,:2]).query_pairs(2*sigmas.max()*np.sqrt(2)))
    for i,j in pairs:
        a,b=blobs[i],blobs[j]
        if a[-1]==b[-1]==0:continue
        # At overlap=0 only the existence of circle intersection matters.
        if np.linalg.norm(a[:2]-b[:2]) < np.sqrt(2)*(a[-1]+b[-1]):
            if a[-1]>b[-1]:b[-1]=0
            else:a[-1]=0
    blobs=blobs[blobs[:,-1]>0]
    sizes=blobs[:,2]*np.sqrt(2)*s*1.3
    xx=x[blobs[:,1].astype(int)];yy=y[blobs[:,0].astype(int)]
    return np.column_stack((xx-sizes,xx+sizes,yy-sizes,yy+sizes))
