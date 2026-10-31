"""Landweber SIRT with positivity, preserving the Tomviz float32 storage path."""
import numpy as np
from .physics_model import parallel_ray
from .preprocessing import NITER,STEP
def reconstruct(tilt_series,angles,iterations=NITER,step=STEP,record_every=10):
    if iterations<0 or int(iterations)!=iterations:raise ValueError('iterations must be nonnegative integer')
    data=np.array(tilt_series,copy=True);angles=np.array(angles,copy=True)
    if np.count_nonzero(angles)<angles.size:angles+=.001
    if np.any(data<0):data-=np.amin(data)
    slices,rays,planes=data.shape
    matrix=parallel_ray(rays,1.,angles,rays,1.);transpose=matrix.T
    volume=np.zeros((slices,rays,rays),dtype='float32',order='F')
    work=np.zeros(rays*rays,dtype='float32');history=[]
    denom=np.linalg.norm(data.astype('float64'))
    for i in range(iterations):
        for s in range(slices):
            b=data[s].T.ravel();work[:]=volume[s].ravel()
            gradient=transpose.dot(b-matrix.dot(work))
            work=work+gradient*step
            volume[s]=work.reshape(rays,rays)
        volume[volume<0]=0
        if record_every and ((i+1)%record_every==0 or i+1==iterations):
            residual=sum(np.sum((matrix.dot(volume[s].ravel())-data[s].T.ravel())**2) for s in range(slices))
            history.append((i+1,float(np.sqrt(residual)/denom)))
    return volume,np.asarray(history,dtype=float).reshape(-1,2)
