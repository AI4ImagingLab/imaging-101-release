"""Nonnegative population inverse and explicit quadratic B-spline operator."""
import numpy as np
from scipy.optimize import minimize
from .physics_model import build_kernel,objective,objective_gradient,forward
from .preprocessing import width_steps
def spline_matrix(count=77,degree=2,n=100):
    if degree==0:return np.eye(count)
    degree=int(np.clip(degree,1,count-1))
    knots=np.r_[np.zeros(degree),np.arange(count-degree+1),np.full(degree,count-degree)]
    u=np.linspace(0,count-degree,n);basis=((u[:,None]>=knots[:-1])&(u[:,None]<knots[1:])).astype(float)
    for k in range(1,degree+1):
        columns=len(knots)-k-1;next_basis=np.zeros((n,columns))
        for j in range(columns):
            left=knots[j+k]-knots[j];right=knots[j+k+1]-knots[j+1]
            if left>0:next_basis[:,j]+=(u-knots[j])/left*basis[:,j]
            if right>0:next_basis[:,j]+=(knots[j+k+1]-u)/right*basis[:,j+1]
        basis=next_basis
    basis[-1]=0;basis[-1,-1]=1
    return basis
def convergence_certificate(weights,kernel,loading):
    """Independent convex first-order and feasible-dual quality checks."""
    x=np.asarray(weights);residual=forward(x,kernel)-loading
    lipschitz=2*np.linalg.norm(kernel,2)**2;gradient=2*kernel@residual
    projected=float(np.max(abs(x-np.maximum(0,x-gradient/lipschitz))))
    dual=residual.copy();at=kernel@dual
    correction=max(0.,float(np.max(-at/kernel.sum(axis=1))))+1e-15
    dual+=correction;primal=float(residual@residual)
    dual_value=-float(dual@dual)-2*float(np.asarray(loading)@dual)
    return dict(projected_kkt=projected,relative_primal_dual_gap=float((primal-dual_value)/max(1.,primal)))

def reconstruct(raw,degree=2,initial=None):
    K=build_kernel(raw['relative_pressure'],raw['kernel_relative_pressure'],raw['kernel_loading'])
    widths=raw['pore_width'];start=np.zeros(len(widths)) if initial is None else np.asarray(initial,dtype=float)
    scale=float(np.mean(raw['loading']**2))
    result=minimize(lambda x:objective(x,K,raw['loading'])/scale,start,jac=lambda x:objective_gradient(x,K,raw['loading'])/scale,method='SLSQP',bounds=[(0,None)]*len(widths),constraints=[{'type':'ineq','fun':lambda x:x}],options={'ftol':1e-15,'maxiter':5000})
    if not result.success:raise RuntimeError('Nonnegative adsorption inversion failed: '+result.message)
    weights=result.x;certificate=convergence_certificate(weights,K,raw['loading'])
    if certificate['projected_kkt']>1e-11 or abs(certificate['relative_primal_dual_gap'])>1e-6:raise RuntimeError('SLSQP reported success without convex convergence: '+str(certificate))
    density=weights/width_steps(widths);S=spline_matrix(len(widths),degree)
    axis=S@widths;smoothed=S@density;cumulative=np.cumsum(smoothed*width_steps(axis))
    return dict(reconstruction=smoothed,bin_weights=weights,cumulative_weights=cumulative,width_axis=axis,unsmoothed_density=density,kernel=K,fitted_loading=forward(weights,K),objective=objective(weights,K,raw['loading']),iterations=int(result.nit),smoothing_matrix=S,convergence_certificate=certificate)
