"""Local split-Bregman TV and Laplacian-penalized LSQR; no domain solver."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr,LinearOperator
from .preprocessing import normalize_clock,rotational_displacement
from .physics_model import accumulate_events,normalized_motion,directional_matrix,derivative_matrices
def normalize_log_image(x):
    low,high=np.percentile(x,[1,99]);width=high-low
    return np.clip((x-low)/width if width else x,0.,1.)
def reconstruct(events,angular_velocity,intrinsics,method='TV',weight=None,outer_iterations=20,warp_fraction=1.0):
    tau,duration=normalize_clock(events);motion=rotational_displacement(angular_velocity,intrinsics,duration)
    iwe=accumulate_events(tau,motion*warp_fraction);ux,uy,mag,mask=normalized_motion(motion)
    b=iwe*mask/mag;A=directional_matrix(uy,ux);Dy,Dx,Lap=derivative_matrices(iwe.shape)
    if method=='Laplacian':
        weight=.3 if weight is None else float(weight)
        op=sparse.vstack((A,weight*Lap),format='csr');rhs=np.concatenate((b.ravel(),np.zeros(b.size)))
        x=lsqr(op,rhs,iter_lim=100)[0];history=[]
    elif method=='TV':
        weight=.1 if weight is None else float(weight);scale=np.sqrt(weight)
        derivatives=(Dy,Dx);op=sparse.vstack((A,scale*Dy,scale*Dx),format='csr')
        x=np.full(b.size,.5);dual=[np.zeros(b.size) for _ in derivatives];split=[np.zeros(b.size) for _ in derivatives];history=[]
        for _ in range(outer_iterations):
            old=x.copy()
            for _ in range(5):
                rhs=np.concatenate([b.ravel()]+[scale*(d-c) for d,c in zip(split,dual)])
                x=x+lsqr(op,rhs-op@x,iter_lim=10,damp=1e-4)[0]
                for k,D in enumerate(derivatives):
                    z=D@x+dual[k];split[k]=np.sign(z)*np.maximum(np.abs(z)-weight,0.)
            for k,D in enumerate(derivatives):dual[k]+=D@x-split[k]
            history.append([.5*np.linalg.norm(b.ravel()-A@x)**2,sum(np.sum(np.abs(D@x)) for D in derivatives),np.linalg.norm(x-old)])
            if history[-1][-1]<=1e-4:break
    else:raise ValueError('Expected TV or Laplacian')
    image=normalize_log_image(x.reshape(iwe.shape))
    return image,dict(raw_log_image=x.reshape(iwe.shape),iwe=iwe,niwe=b,displacement=motion,unit_flowx=ux,unit_flowy=uy,flow_magnitude=mag,history=np.asarray(history),A=A)
