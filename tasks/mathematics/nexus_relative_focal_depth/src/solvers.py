"""Explicit local linearized ADMM; GPL-2.0 source extraction."""
import numpy as np
from .physics_model import gradient,adjoint,solve_neumann,neumann_eigenvalues,polynomial_value,objective
from .preprocessing import modified_laplacian,fit_focus_polynomials,initial_focal_coordinate


def shrink(x,y,threshold):
    """Pointwise isotropic two-vector soft threshold, including exact zero."""
    length=np.sqrt(x*x+y*y)
    factor=np.zeros_like(length)
    np.divide(np.maximum(length-threshold,0),length,out=factor,where=length>0)
    return factor*x,factor*y


def linearized_admm(derivative,initial,iterations=400,data_fidelity=6.,descent_step=None,
                    penalty=1.,capture_steps=(1,2,10,100,400),callback=None):
    """Native400-step source path; no convergence/accuracy claim is inferred.

    Derivative is of NEGATIVE contrast. The u linear solve is followed by
    source pointwise clipping; this safeguard is not an exact constrained
    quadratic proximal operator. Penalty grows after zero-based iteration>0;
    the scaled dual is rescaled by1/1.02 after each such growth.
    """
    tau=np.float32(8/data_fidelity if descent_step is None else descent_step)
    data_fidelity=np.float32(data_fidelity);lam=np.float32(penalty)
    u=np.array(initial,dtype=np.float32,copy=True)
    gx=np.zeros_like(u);gy=np.zeros_like(u);bx=np.zeros_like(u);by=np.zeros_like(u)
    eigenvalues=neumann_eigenvalues(u.shape,np.float32);snapshots={}
    for it in range(iterations):
        # Separate scalar products and sums preserve original launch order.
        vx=gx-bx;vy=gy-by
        dx=np.zeros_like(u);dy=np.zeros_like(u)
        dx[:,0]=-vx[:,0];dx[:,1:]=vx[:,:-1]-vx[:,1:]
        dy[0]=-vy[0];dy[1:]=vy[:-1]-vy[1:]
        dx*=lam;dy*=lam
        data_step=polynomial_value(derivative,u)*np.float32(-tau*data_fidelity)
        rhs=u+((dx+dy)+data_step)
        u=solve_neumann(rhs,lam,eigenvalues)
        np.clip(u,-10,10,out=u)
        ux,uy=gradient(u)
        gx,gy=shrink(ux+bx,uy+by,np.float32(tau/lam))
        bx+=ux-gx;by+=uy-gy
        if it+1 in capture_steps:
            snapshots[it+1]={k:v.copy() for k,v in [('u',u),('gx',gx),('gy',gy),('bx',bx),('by',by),('ux',ux),('uy',uy)]}
        if it>0:
            lam*=np.float32(1.02)
            bx*=np.float32(1)/np.float32(1.02);by*=np.float32(1)/np.float32(1.02)
        if callback is not None:callback(it+1,u)
    return u,snapshots


def reconstruct(observations,iterations=400,data_fidelity=6.,index_offset=1,callback=None):
    """Native uint8[F,H,W,3] -> field and identified fresh intermediate arrays."""
    sharpness=modified_laplacian(observations)
    coefficients,derivative=fit_focus_polynomials(sharpness)
    initial=initial_focal_coordinate(sharpness,index_offset=index_offset)
    result,steps=linearized_admm(derivative,initial,iterations=iterations,data_fidelity=data_fidelity,callback=callback)
    return result,{'initial':initial,'coefficients':coefficients,'derivative':derivative,
                   'steps':steps,'sharpness':sharpness,'objective':objective(coefficients,result,data_fidelity)}


def classical_focus(sharpness,window=9,median_width=1):
    """Computed classical comparison, with explicitly specified box/median sizes.

    General separable filters suffice here: these comparison estimators do not
    assert CUDA tap-by-tap parity for the selected method's initializer.
    """
    from scipy.ndimage import uniform_filter,median_filter
    smoothed=uniform_filter(sharpness,size=(1,window,window),mode='nearest')
    field=np.float32(20)*(np.argmax(smoothed,axis=0).astype(np.float32)/np.float32(len(sharpness)-1)-np.float32(.5))
    if median_width>1:field=median_filter(field,size=median_width,mode='nearest')
    return field


def tv_denoise_only(observation,weight,iterations=250):
    """Independent explicit primal-dual ROF comparison: .5||u-z||²+weightTV(u).

    tau=sigma=.35 gives tau*sigma*||K||²<=.98<1; theta=1. The
    pointwise quadratic proximal step and dual Euclidean-ball projection are
    explicit. This is a comparison, not a replacement focus-stack inverse.
    """
    z=np.asarray(observation,np.float32);u=z.copy();bar=u.copy()
    px=np.zeros_like(z);py=np.zeros_like(z);tau=np.float32(.35);sigma=np.float32(.35)
    for _ in range(iterations):
        dx,dy=gradient(bar);px+=sigma*dx;py+=sigma*dy
        scale=np.maximum(np.float32(1),np.hypot(px,py)/np.float32(weight))
        px/=scale;py/=scale
        old=u
        u=(u-tau*adjoint(px,py)+tau*z)/(np.float32(1)+tau)
        bar=np.float32(2)*u-old
    return u
