"""GPL-3.0 Tobias Meissner2024 planar MURA and Accorsi decoding, locally extracted."""
import numpy as np
from scipy.interpolate import interpn
from .preprocessing import preprocess
from .physics_model import decoding_pattern,projected_size,nearest_resize,central_crop,cyclic_convolve,PITCH_MM,MASK_TO_DETECTOR_MM,SOURCE_PLANE_MM
def reconstruct(raw,mask,distance=SOURCE_PLANE_MM,sigma=1.,reject_outliers=True):
    """Return float32(N,N) signed source response in summed keV units, N=126 at50mm.

    Direct linear decoding after the published preprocessing. No iterative solver,
    arbitrary peak normalization, nonnegativity clipping or rotation is performed.
    """
    y=preprocess(raw,sigma,reject_outliers);n=projected_size(distance)
    if n>y.shape[0]:raise ValueError('Projected central mask exceeds detector')
    d=nearest_resize(decoding_pattern(mask),n);c=central_crop(y,n/y.shape[0])
    return cyclic_convolve(c,d).astype(np.float32)
def accorsi(raw,mask,distance=SOURCE_PLANE_MM,sigma=1.,reject_outliers=True):
    """Comparison only: original sparse/delta decoding, float32(K,K), K124 at50mm."""
    y=preprocess(raw,sigma,reject_outliers);d=decoding_pattern(mask);alpha=(1+MASK_TO_DETECTOR_MM/distance)*.08/PITCH_MM;k=int(np.round(alpha))
    sparse=np.zeros((k*d.shape[0],k*d.shape[1]),np.float32);sparse[::k,::k]=d
    xq=np.linspace(-alpha*d.shape[1]/2,alpha*d.shape[1]/2,sparse.shape[1]);yq=np.linspace(-alpha*d.shape[0]/2,alpha*d.shape[0]/2,sparse.shape[0]);X,Y=np.meshgrid(xq,yq)
    points=np.array([X.flatten(),Y.flatten()]).T
    gridx=np.linspace(-y.shape[0]/2-1,y.shape[0]/2+1,y.shape[0]);gridy=np.linspace(-y.shape[1]/2-1,y.shape[1]/2+1,y.shape[1])
    sample=interpn((gridx,gridy),y,xi=points,method='linear',bounds_error=False).reshape(sparse.shape).T
    return cyclic_convolve(sample,sparse).astype(np.float32)
