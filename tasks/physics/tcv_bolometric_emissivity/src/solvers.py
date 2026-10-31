"""Chambolle-Dossal accelerated projected gradient, original offline MAP."""
import numpy as np
from .preprocessing import prepare,SAMPLING
from .physics_model import gradient_matrix,diffusion_tensor,diffusion_matrix

def reconstruct_frame(matrix,values,flux,mask,alpha=.01,regularization=10.,sigma=.025,max_iterations=10000,tolerance=1e-5,positive=False,return_history=False):
    """Return absolute-scale frame and optional normalized objective trace."""
    scale=float(values.max())
    if scale<=0: raise ValueError('positive frame normalization required')
    y=values/scale
    a=matrix
    lap=diffusion_matrix(diffusion_tensor(flux,alpha),gradient_matrix(flux.shape))
    # Same conservative derivative Lipschitz bound used by the original Pyxu operator.
    spectral=float(np.linalg.eigvalsh(a@a.T)[-1])
    step=1/(spectral/sigma**2+regularization*16/SAMPLING**2)
    x=(a.T@y).reshape(flux.shape)
    prev=x.copy(); history=[]
    for k in range(max_iterations):
        momentum=k/(k+4.)
        extrap=x+momentum*(x-prev)
        flat=extrap.ravel()
        grad=(a.T@(a@flat-y))/sigma**2+regularization*(lap@flat)
        new=(extrap-step*grad.reshape(x.shape))*mask
        if positive: new=np.maximum(new,0)
        relative=np.linalg.norm(new-x)/max(np.linalg.norm(x),np.finfo(float).tiny)
        if return_history:
            v=new.ravel(); r=a@v-y
            history.append((float(r@r/(2*sigma**2)+regularization*v@(lap@v)/2),float(relative)))
        prev,x=x,new
        if relative<=tolerance: break
    return (x*scale,np.array(history)) if return_history else x*scale

def reconstruct(data,equilibrium='measured',alpha=.01,indices=None,positive=False):
    """Reconstruct every native time by default; subsets are explicit comparisons."""
    matrix,values,flux,mask=prepare(data,equilibrium)
    ids=np.arange(len(values)) if indices is None else np.asarray(indices)
    return np.array([reconstruct_frame(matrix,values[i],flux[i],mask,alpha=alpha,positive=positive) for i in ids])
