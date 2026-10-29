"""HiFRAP variable projection with the original trust-region update settings."""
import numpy as np
from scipy.optimize import least_squares
from .preprocessing import prepare,PIXEL_SIZE,TIME_STEP
from .physics_model import compression_matrix,profile_residual

def reconstruct(data,model='diffusion',nq=9,ignore_bleaching=False):
    """Return fresh fit, parameters and diagnostics from the complete movie.

    Optional reaction and reaction-diffusion models are notebook comparisons.
    No file writes; all field values retain the observed camera scale.
    """
    p=prepare(data); y=p['normalized']; nt,nx=y.shape
    c=compression_matrix(nx,nq)
    yq=y@c.T
    eps=0. if ignore_bleaching else p['epsilon']
    d0=(nx/2)**2/((nt-1)/2)/8
    b0=np.log(2)/((nt-1)/2)
    db=[(nx/nq)**2/nt/40,nx**2/5]; bb=[-3/nt,100/nt]
    if model=='diffusion': initial=[d0]; bounds=db
    elif model=='reaction': initial=[b0]; bounds=bb
    elif model=='reaction-diffusion': initial=[d0,b0]; bounds=([db[0],bb[0]],[db[1],bb[1]])
    else: raise ValueError('Unknown model')
    def unpack(z):
        if model=='diffusion': return z[0],0.
        if model=='reaction': return 0.,z[0]
        return z
    def objective(z):
        d,b=unpack(z)
        return profile_residual(d,yq,eps,p['background'],p['stationary'],model,b)
    fit=least_squares(objective,initial,bounds=bounds,x_scale=np.abs(initial),method='trf',xtol=1e-4,ftol=1e-12,gtol=1e-12,loss='linear',diff_step=1e-2,jac='2-point',max_nfev=500)
    d,b=unpack(fit.x)
    residual,cost,trace=profile_residual(d,yq,eps,p['background'],p['stationary'],model,b,True)
    fitted_q=yq-residual*np.sqrt(trace)
    field=(fitted_q@c)*p['norm'][1]+p['norm'][0]
    covariance=np.linalg.pinv(fit.jac.T@fit.jac)*cost/trace
    centered=yq.copy();centered[:,0]-=centered[:,0].mean()
    n_theta=1 if model=='diffusion' else 2
    r2=1-(cost*trace/(trace-n_theta))/np.sum(centered**2/(yq.size-1))
    return dict(reconstruction=field[None].astype(np.float64),diffusivity=np.array([[d*PIXEL_SIZE**2/TIME_STEP]],dtype=np.float64),
                loss_rate=b/TIME_STEP,epsilon=eps,cost=cost,trace=trace,r2=r2,covariance=covariance,
                compressed=yq,residual=residual,preprocessing=p,nfev=fit.nfev,success=bool(fit.success))
