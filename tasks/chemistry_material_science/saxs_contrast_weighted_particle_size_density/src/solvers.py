"""Explicit Irena interior-point gradient inverse; no domain packages."""
import numpy as np
from .preprocessing import estimate_background,radius_grid,bin_widths
from .physics_model import build_kernel
def interior_point_gradient(kernel,target,error,max_iter=1000,approach=0.95):
    m,n=kernel.shape;A=kernel/error[:,None];b=target/error
    ata=A.T@A;atb=A.T@b;x=np.full(n,1e-32);history=[]
    for iteration in range(1,max_iter+1):
        atax=ata@x;gradient=atax-atb;direction=-(x/np.maximum(atax,1e-300))*gradient
        atp=ata@direction;denominator=float(direction@atp)
        if abs(denominator)<1e-300:break
        alpha=max(-float(direction@gradient)/denominator,0.)
        for i in range(n):
            if direction[i]<0:alpha=min(alpha,approach*(-x[i]/direction[i]))
        x=np.maximum(x+alpha*direction,1e-300)
        residual=(target-kernel@x)/error;chi2=float(residual@residual);history.append(chi2)
        if chi2/m<=1.:break
    return dict(bin_mass=x,iterations=iteration,history=np.asarray(history))
def reconstruct(q,intensity,error,n_bins=500,background=None,max_iter=1000):
    r=radius_grid(n_bins);w=bin_widths(r);G=build_kernel(q,r)
    B=estimate_background(q,intensity) if background is None else float(background)
    fit=interior_point_gradient(G,intensity-B,error,max_iter=max_iter)
    fit.update(reconstruction=fit['bin_mass']/np.maximum(w,1e-300),background=B,radius=r,widths=w,kernel=G)
    fit['model']=G@fit['bin_mass']+B;fit['residual']=(intensity-fit['model'])/error
    return fit
