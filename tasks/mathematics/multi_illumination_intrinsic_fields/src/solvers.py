"""Explicit published median-gradient separation, masked integration and IRLS."""
import time
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator,cg
from .physics_model import gradients,normal_equations
def integrate_gradients(ty,tx,mask,wy=None,wx=None,rtol=1e-11,maxiter=20000):
    """CG solves the source least-squares equations; mean-log gauge fixes nullspace.

    Adds only the constant projector to the operator. This does not change
    any gradient normal equation; the returned foreground log-field has zero
    mean. Jacobi preconditioning, zero start, explicit convergence failure.
    """
    A,b,numbers=normal_equations(ty,tx,mask,wy,wx)
    if connected_components(A,directed=False,return_labels=False)!=1:raise ValueError('Mask must be connected')
    n=len(b);b=b-b.mean();diagonal=A.diagonal()+1/n
    operator=LinearOperator(A.shape,matvec=lambda v:A@v+v.mean(),dtype=np.float64)
    preconditioner=LinearOperator(A.shape,matvec=lambda v:v/diagonal,dtype=np.float64)
    history=[]
    def observe(v):
        history.append(None)
    t=time.perf_counter();u,status=cg(operator,b,x0=np.zeros(n),rtol=rtol,atol=0.,maxiter=maxiter,M=preconditioner,callback=observe)
    if status:raise RuntimeError(f'Gradient integration CG did not converge: {status}')
    u-=u.mean();image=np.zeros(mask.shape);image[mask]=u
    residual=float(np.linalg.norm(A@u-b)/(np.linalg.norm(b) or 1.))
    return image,{'converged':True,'iterations':len(history),'relative_normal_residual':residual,'seconds':time.perf_counter()-t,'rtol':rtol}
def median_targets(multiple):
    """Source floor3 before logarithm; median over ten native illuminations."""
    gy,gx=gradients(np.log(np.maximum(np.asarray(multiple,dtype=np.float64),3.)))
    return np.median(gy,axis=-1),np.median(gx,axis=-1)
def estimate(image,multiple,mask,method='weiss',threshold=.1,l1=False):
    """Return foreground S,R plus integration diagnostics, without hidden assets.

    Primary is parameter-free Weiss. Optional explicitly named controls retain
    source threshold strictness and 20 IRLS updates/1e-4 residual floor.
    """
    image=np.asarray(image,dtype=np.float64);m=np.asarray(mask,dtype=bool)
    if method in ('weiss','weiss_retinex'):ty,tx=median_targets(multiple)
    elif method=='gray_retinex':
        image=np.maximum(image,3.);ty,tx=gradients(np.where(m,np.log(image),0.))
    elif method=='color_retinex':
        rgb=np.maximum(image,3.);gy,gx=gradients(np.log(rgb));gray_y=gy.mean(axis=2);gray_x=gx.mean(axis=2)
        match_y=(np.linalg.norm(gy-gray_y[...,None],axis=2)>threshold)|(abs(gray_y)>threshold)
        match_x=(np.linalg.norm(gx-gray_x[...,None],axis=2)>threshold)|(abs(gray_x)>threshold)
        image=np.maximum(rgb.mean(axis=2),3.);ty,tx=gradients(np.log(image));ty*=match_y;tx*=match_x
    else:raise ValueError('Unknown intrinsic method')
    if method in ('weiss_retinex','gray_retinex'):ty=ty*(abs(ty)>threshold);tx=tx*(abs(tx)>threshold)
    def losses(u):
        gy,gx=gradients(u);ey=m[1:]&m[:-1];ex=m[:,1:]&m[:,:-1]
        residual=np.concatenate([(gy-ty)[1:][ey],(gx-tx)[:,1:][ex]])
        return {'gradient_l1':float(abs(residual).sum()),'gradient_l2_squared':float(residual@residual)}
    u,record=integrate_gradients(ty,tx,m);record.update(losses(u));records=[record]
    if l1:
        for _ in range(20):
            gy,gx=gradients(u)
            wy=1/(2*np.maximum(m*abs(gy-ty),1e-4));wx=1/(2*np.maximum(m*abs(gx-tx),1e-4))
            u,record=integrate_gradients(ty,tx,m,wy,wx);record.update(losses(u));records.append(record)
    reflectance=np.zeros(m.shape);reflectance[m]=np.exp(u[m]);shading=np.zeros(m.shape);shading[m]=image[m]/reflectance[m]
    return {'shading':shading,'reflectance':reflectance,'integration':records,'method':method,'l1':l1}
