"""Float32 background and separable Gaussian mathematics.

AccPbFRET routines adapted from Roszik/Szollosi/Vereb, GPL-3.0-or-later.
Gaussian discretization follows public-domain ImageJ1.54p GaussianBlur.
"""
import math
import numpy as np

BACKGROUND_ROI=(0,64,448,512)
GAUSSIAN_SIGMA=1.0
GAUSSIAN_ACCURACY=.01
SIGNAL_THRESHOLD=100.0

def subtract_background(image,roi=BACKGROUND_ROI):
    """Subtract the ROI float32 mean, clamp negative counts; return image, mean."""
    a=np.asarray(image,dtype=np.float32)
    y0,y1,x0,x1=roi
    level=np.float32(np.mean(a[y0:y1,x0:x1],dtype=np.float64))
    return np.maximum(a-level,np.float32(0)),level

def gaussian_half_kernel(sigma=GAUSSIAN_SIGMA,accuracy=GAUSSIAN_ACCURACY,max_radius=512):
    """ImageJ finite Gaussian with smooth quadratic edge and tail mass."""
    if sigma<=0 or not 0<accuracy<.02:raise ValueError('sigma>0 and 0<accuracy<.02 required')
    limit=max(50,max_radius)
    radius=min(int(math.ceil(sigma*math.sqrt(-2*math.log(accuracy))))+1,limit)
    k=np.array([math.exp(-.5*i*i/sigma/sigma) for i in range(radius)],dtype=np.float32)
    if radius<limit and radius>3:
        slope=float('inf');r=radius
        while r>radius//2:
            r-=1;a=math.sqrt(float(k[r]))/(radius-r)
            if a<slope:slope=a
            else:break
        for j in range(r+2,radius):k[j]=np.float32((radius-j)**2*slope*slope)
    total=float(k[0])+sum(2*float(v) for v in k[1:]) if radius<limit else sigma*math.sqrt(2*math.pi)
    tail=np.empty_like(k);rsum=.5+.5*float(k[0])/total
    for i in range(radius):
        v=float(k[i])/total;k[i]=np.float32(v);rsum-=v;tail[i]=np.float32(rsum)
    return k,tail

def _convolve_axis(image,kernel,tail,axis):
    """Symmetric pair accumulation and explicit nearest-edge tail arithmetic."""
    a=np.moveaxis(np.asarray(image,dtype=np.float32),axis,-1)
    n=a.shape[-1];r=len(kernel)
    if n<=2*r:raise ValueError('line too short for selected Gaussian discretization')
    result=np.empty_like(a)
    interior=a[...,r:n-r]*kernel[0]
    for j in range(1,r):interior+=kernel[j]*(a[...,r-j:n-r-j]+a[...,r+j:n-r+j])
    result[...,r:n-r]=interior
    for i in list(range(r))+list(range(n-r,n)):
        value=a[...,i]*kernel[0]
        if i<r:value+=tail[i]*a[...,0]
        if i+r>=n:value+=tail[n-i-1]*a[...,-1]
        for j in range(1,r):
            pair=np.zeros(a.shape[:-1],dtype=np.float32)
            if i-j>=0:pair+=a[...,i-j]
            if i+j<n:pair+=a[...,i+j]
            value+=kernel[j]*pair
        result[...,i]=value
    return np.moveaxis(result,-1,axis)

def gaussian_blur(image,sigma=GAUSSIAN_SIGMA):
    """Full-frame horizontal then vertical float32 smoothing, ADU units."""
    if sigma==0:return np.asarray(image,dtype=np.float32).copy()
    a=np.asarray(image,dtype=np.float32)
    k,t=gaussian_half_kernel(sigma,max_radius=max(a.shape))
    return _convolve_axis(_convolve_axis(a,k,t,1),k,t,0)

def registration_shift(before,after):
    """Integer circular cross-correlation peak; zero shift on bound acquisition."""
    a=np.asarray(before,dtype=np.float64);b=np.asarray(after,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('matched two dimensional images required')
    c=np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(b))).real
    # Source traverses x before y and uses its first strictly largest peak.
    x,y=np.unravel_index(np.argmax(c.T),c.T.shape)
    return (int(y if y<=a.shape[0]//2 else y-a.shape[0]),int(x if x<=a.shape[1]//2 else x-a.shape[1]))

def translate_zero(image,shift):
    """Native integer shift with zero-filled exposed rows/columns."""
    a=np.asarray(image);dy,dx=map(int,shift);out=np.zeros_like(a)
    h,w=a.shape[-2:]
    if abs(dy)>=h or abs(dx)>=w:return out
    out[...,max(dy,0):min(h,h+dy),max(dx,0):min(w,w+dx)]=a[...,max(-dy,0):min(h,h-dy),max(-dx,0):min(w,w-dx)]
    return out

def preprocess_sample(observations,sigma=GAUSSIAN_SIGMA):
    """TCYX raw (2,2,Y,X) -> TCYX background and Gaussian stages and offsets."""
    a=np.asarray(observations)
    if a.ndim!=4 or a.shape[:2]!=(2,2) or not np.isfinite(a).all():raise ValueError('finite TCYX (2,2,Y,X) required')
    background=np.empty(a.shape,dtype=np.float32);smooth=np.empty_like(background);levels=np.empty((2,2),np.float32)
    for t in range(2):
        for c in range(2):
            background[t,c],levels[t,c]=subtract_background(a[t,c]);smooth[t,c]=gaussian_blur(background[t,c],sigma)
    return {'background':background,'smooth':smooth,'offsets':levels}
