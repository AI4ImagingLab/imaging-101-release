"""GPL-3.0 adapted Tobias Meissner2024: explicit mask and shadow mathematics.

No TensorFlow, Pillow or external reconstruction package is used.
"""
import numpy as np
DETECTOR_PIXELS=256;DETECTOR_MM=14.08;MASK_MM=9.92;MASK_TO_DETECTOR_MM=20.;SOURCE_PLANE_MM=50.
PITCH_MM=DETECTOR_MM/DETECTOR_PIXELS
def central_crop(image,fraction):
    """Original integer-truncated centred crop; fraction in[0,1]."""
    a=np.asarray(image);f=min(max(float(fraction),0.),1.);h,w=a.shape
    return a[int(h/2-h*f/2):int(h/2+h*f/2),int(w/2-w*f/2):int(w/2+w*f/2)]
def decoding_pattern(mask):
    """Convert supplied NTHT mask to signed central62² MURA decoder."""
    d=central_crop(mask,.5).copy()
    if d[-1,:].sum()==0:d=np.roll(d,1,axis=0)
    if d[:,-1].sum()==0:d=np.roll(d,1,axis=1)
    q=d[1::2,1::2].copy();q[q.shape[0]//2,q.shape[1]//2]=0.;q[q==0]=-1.;d[1::2,1::2]=q
    return np.roll(d,[d.shape[0]//2,d.shape[1]//2],axis=[0,1])
def projected_size(distance=SOURCE_PLANE_MM):
    """Central pattern side in pixels; geometric magnification1+b/z."""
    if distance<=0:raise ValueError('Positive source distance required')
    return int(np.round((1+MASK_TO_DETECTOR_MM/distance)*MASK_MM/PITCH_MM/2))
def nearest_resize(a,size):
    """Centre-sampled nearest-neighbour resize matching Pillow NEAREST exactly."""
    h,w=a.shape
    i=np.minimum(np.floor((np.arange(size)+.5)*h/size).astype(int),h-1)
    j=np.minimum(np.floor((np.arange(size)+.5)*w/size).astype(int),w-1)
    return a[i[:,None],j[None,:]]
def cyclic_convolve(source,shadow):
    """Circular discrete convolution with backward-normalized FFT; sum not mean."""
    return np.fft.ifft2(np.fft.fft2(source)*np.fft.fft2(shadow)).real
def cyclic_adjoint(detector,shadow):
    """Exact Euclidean adjoint of source->cyclic_convolve(source,shadow)."""
    return np.fft.ifft2(np.fft.fft2(detector)*np.conj(np.fft.fft2(shadow))).real
def source_coordinates(size=126,distance=SOURCE_PLANE_MM):
    """Nominal source-plane coordinates in mm, unrotated author output convention."""
    return -(np.arange(size)-size/2)*PITCH_MM*distance/MASK_TO_DETECTOR_MM
