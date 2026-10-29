"""GPL-3.0 extraction, separately qualified Eq19 Newton reproduction.

Corrections relative to author source are explicit: exact objective gradient,
one prior curvature, correct mixed Airy derivative and additive Newton step.
Stable direct QR, absolute-spacing noise units and a fixed dimensionless
coordinate scaling with a disclosed SVD rank convention qualify the numerics.
"""
import time
import numpy as np
from scipy.linalg import svd,solve_triangular,qr
from .physics_model import place_bases,density_basis,strain_basis,traction_basis,acquisition_operators,prediction,jacobian

def initialize(basis,grid_points,nominal_spacing=4.056):
    d=density_basis(grid_points,basis);variance=basis['sd'];n=d.shape[1]
    gamma=np.vstack((d/1e-8,np.diag(1/np.sqrt(variance))));q,r=qr(gamma,mode='economic')
    rhs=np.r_[np.full(len(d),nominal_spacing/1e-8),np.zeros(n)]
    wd=solve_triangular(r,q.T@rhs)
    return np.r_[wd,np.zeros(len(basis['lcx']))]

def objective(weights,mean,prior,linear,bilinear,traction,observations,noise_sd=4.056e-4,traction_sd=1e-6):
    m=linear.shape[1];r=observations-prediction(weights,linear,bilinear);t=traction@weights[m:];v=weights-mean
    return float(.5*(r@r/noise_sd**2+t@t/traction_sd**2+np.sum(v*v/prior)))

def derivatives(weights,mean,prior,linear,bilinear,traction,observations,noise_sd=4.056e-4,traction_sd=1e-6):
    """Return negative-objective gradient direction g and objective Hessian H."""
    m=linear.shape[1];j=jacobian(weights,linear,bilinear);nt=len(traction)
    full=np.vstack((j,np.c_[np.zeros((nt,m)),traction]));res=np.r_[observations-prediction(weights,linear,bilinear),-traction@weights[m:]]
    inverse_variance=np.r_[np.full(len(observations),1/noise_sd**2),np.full(nt,1/traction_sd**2)]
    g=full.T@(inverse_variance*res)-(weights-mean)/prior
    h=(full.T*inverse_variance)@full+np.diag(1/prior)
    mixed=np.einsum('n,njk->jk',res[:len(observations)]/noise_sd**2,bilinear,optimize=True)
    h[:m,m:]-=mixed;h[m:,:m]-=mixed.T
    return g,h,full

def newton_direction(gradient,hessian,coordinate_scale,cutoff=1e-12):
    """Congruence-scaled truncated-SVD direction, returned in physical coefficients."""
    scaled=coordinate_scale[:,None]*hessian*coordinate_scale[None,:]
    u,s,vh=svd(scaled,lapack_driver='gesvd');keep=s>cutoff*s.max()
    return coordinate_scale*(vh[keep].T@((u[:,keep].T@(coordinate_scale*gradient))/s[keep])),s,int(np.sum(keep))

def solve_weights(linear,bilinear,traction,observations,mean,prior,max_iterations=15,noise_sd=4.056e-4,traction_sd=1e-6,svd_cutoff=1e-12):
    x=mean.copy();cost=objective(x,mean,prior,linear,bilinear,traction,observations,noise_sd,traction_sd);xs=[x.copy()];history=[(0,cost,0.,0.)];first=None;alpha_saved=1.
    for iteration in range(1,max_iterations+1):
        g,h,j=derivatives(x,mean,prior,linear,bilinear,traction,observations,noise_sd,traction_sd)
        if iteration==1:coordinate_scale=1/np.sqrt(np.diag(h))
        direction,s,rank=newton_direction(g,h,coordinate_scale,svd_cutoff)
        if first is None:first={'gradient':g.copy(),'hessian':h.copy(),'jacobian':j.copy(),'direction':direction.copy(),'coordinate_scale':coordinate_scale.copy(),'singular_values':s.copy(),'rank':rank}
        # Source custom bidirectional search preserved; each proposal is corrected
        # additive Newton and accepted only by the actual Eq19 objective.
        best=np.inf;found=False;forward=True;forward_steps=0;backward_steps=0;alpha=alpha_saved;accepted=None
        for k in range(53):
            candidate=x+alpha*direction
            value=objective(candidate,mean,prior,linear,bilinear,traction,observations,noise_sd,traction_sd)
            if value<cost or found:
                found=True
                if value<best:
                    alpha_saved=alpha;best=value;accepted=candidate
                elif forward_steps>1 or backward_steps>0:break
                else:forward=False;alpha=alpha_saved
                if forward:
                    if alpha==1:break
                    alpha*=1.5;forward_steps+=1
                else:alpha/=2;backward_steps+=1
            else:forward=False;alpha/=2
        if accepted is None:break
        old_cost=cost;old=x;x=accepted;cost=best;xs.append(x.copy());history.append((iteration,cost,alpha_saved,float(np.linalg.norm(x-old))))
        if abs(cost-old_cost)<1e-8*abs(cost) or k==52:break
    return x,{'weights_path':np.array(xs).T,'history':np.array(history),'first':first,'mean':mean,'prior':prior}

def reconstruct(data,max_iterations=15,noise_sd=4.056e-4,traction_sd=1e-6,counts=(15,10),svd_cutoff=1e-12):
    """Actively estimate all four fields from native1798 line averages."""
    start=time.perf_counter();b=place_bases(data['direction'],counts=counts);linear,bilinear=acquisition_operators(data,b)
    points=np.c_[data['grid_x'].ravel(order='F'),data['grid_y'].ravel(order='F')];tb=traction_basis(data['traction_points'],data['traction_normals'],b);mu=initialize(b,points);prior=np.r_[b['sd'],b['sc']]
    weights,diagnostics=solve_weights(linear,bilinear,tb,data['observations'],mu,prior,max_iterations,noise_sd,traction_sd,svd_cutoff)
    m=len(b['ldx']);spacing=(density_basis(points,b)@weights[:m]).reshape(data['grid_x'].shape,order='F');eps=strain_basis(points,b)@weights[m:]
    strain=np.stack([eps[c::3].reshape(data['grid_x'].shape,order='F') for c in range(3)])
    diagnostics.update(weights=weights,basis=b,prediction=prediction(weights,linear,bilinear),elapsed=time.perf_counter()-start,traction=(tb@weights[m:]).reshape(-1,2))
    return {'strain':strain[None],'lattice_spacing':spacing[None]},diagnostics
