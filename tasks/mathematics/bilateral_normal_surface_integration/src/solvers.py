"""Jacobi-CG and bilateral updates; GPL-3.0 adaptation, Xu Cao (2022)."""
import numpy as np
from scipy.sparse import diags,spdiags
from scipy.sparse.linalg import cg
from .physics_model import system,weights


def integrate(normal,mask,k=2.,step_size=1.,max_iter=100,tol=1e-4,cg_max_iter=5000,cg_tol=1e-3):
    """Orthographic normal integration; return zero-mean height and diagnostics.

    No depth prior, perspective model or hidden surface is used in this bound
    task. Missing operator rows and their original RHS terms are preserved.
    Outside support the explicit output convention is zero.
    """
    a,b,ops=system(normal,mask,step_size);count=int(mask.sum())
    w=.5*np.ones(4*count);z=np.zeros(count);energy=.5*np.dot(b,b)
    history=[];cg_info=[]
    for _ in range(max_iter):
        W=diags(w);matrix=(a.T@W@a).tocsr();rhs=a.T@(W@b)
        pre=spdiags(1/np.clip(matrix.diagonal(),1e-5,None),0,count,count)
        z,info=cg(matrix,rhs,x0=z,M=pre,maxiter=cg_max_iter,rtol=cg_tol)
        cg_info.append(int(info))
        if info!=0:raise RuntimeError('Inner CG did not converge: '+str(info))
        w,horizontal,vertical=weights(z,ops,k)
        old=energy;residual=a@z-b;energy=float(residual@(w*residual));history.append(energy)
        # Source's 0/0 relative test is undefined for a perfectly flat field.
        # Define the zero-residual stationary case explicitly; bound inputs
        # have strictly positive energies, so their path is unchanged.
        if (old==0 and energy==0) or (old>0 and abs(energy-old)/old<tol):break
    if not history:raise ValueError('At least one iteration required')
    height=np.zeros(mask.shape);height[mask]=z-z.mean()
    h=np.zeros(mask.shape);v=np.zeros(mask.shape);h[mask]=horizontal;v[mask]=vertical
    return height,{'energy':np.asarray(history),'horizontal_weights':h,'vertical_weights':v,'iterations':len(history),'cg_info':np.asarray(cg_info),'original_gauge':float(z.mean()),'weighted_residual_rms':float(np.sqrt(energy/(2*count)))}


def reconstruct(data):
    """Reconstruct both full native grids with fixed author-example settings."""
    output={};diagnostics={}
    for name,key,k in [('vase','reconstruction',4.),('tent','tent_reconstruction',1.)]:
        height,diag=integrate(*data[name],k=k);output[key]=height[None];diagnostics[name]=diag
    return output,diagnostics
