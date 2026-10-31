"""Explicit constrained SOLA with fully reorthogonalized Lanczos steps.

SOLApack adaptation; see LICENSE/NOTICE. General BLAS operations and banded
linear solve replace no acquisition model, target, inverse step or constraint.
"""
import numpy as np
from scipy.linalg.blas import ddot,daxpy,dnrm2
from scipy.linalg import solve_banded
from .preprocessing import radial_target_widths
from .physics_model import RotationOperator,ConstrainedOperator,initial_target,gaussian_targets

def modified_gram_schmidt(basis,vector,count):
    """Native sequential single-pass modified Gram–Schmidt, in place."""
    for j in range(count):
        scalar=ddot(basis[:,j],vector)
        vector=daxpy(basis[:,j],vector,a=-scalar)
    return vector

def lanczos_bidiagonalize(operator,start,iterations=1000,progress=False):
    """Golub–Kahan lower bidiagonalization with native full reorthogonalization.

    Return U (grid,iterations+1), V (modes-1,iterations), diagonal and
    subdiagonal. No previously computed factorization is read.
    """
    m=operator.op.np;n=operator.op.nm-1;k=iterations
    U=np.empty((m,k+1),dtype=np.float64,order='F');V=np.empty((n,k),dtype=np.float64,order='F')
    alpha=np.empty(k);beta=np.empty(k)
    U[:,0]=start/dnrm2(start)
    for j in range(k):
        vector=np.array(operator.adjoint(U[:,j]),dtype=np.float64,order='F')
        if j:vector-=beta[j-1]*V[:,j-1]
        vector=modified_gram_schmidt(V,vector,j);alpha[j]=dnrm2(vector)
        if alpha[j]==0:raise ValueError(f'Exact Lanczos breakdown at {j}')
        V[:,j]=vector/alpha[j]
        vector=np.array(operator.forward(V[:,j])-alpha[j]*U[:,j],dtype=np.float64,order='F')
        vector=modified_gram_schmidt(U,vector,j+1);beta[j]=dnrm2(vector)
        if beta[j]==0:raise ValueError(f'Exact Lanczos breakdown at {j}')
        U[:,j+1]=vector/beta[j]
        if progress and ((j+1)%100==0 or j==0):print(f'Lanczos {j+1}/{k}',flush=True)
    return U,V,alpha,beta

def solve_projected(alpha,beta,projected_targets,tradeoff=3.):
    """Explicit B.T B + tradeoff I system for lower bidiagonal B."""
    k=len(alpha);ab=np.zeros((3,k))
    ab[1]=alpha**2+beta**2+tradeoff
    off=beta[:-1]*alpha[1:];ab[0,1:]=off;ab[2,:-1]=off
    rhs=alpha[:,None]*projected_targets[:-1]+beta[:,None]*projected_targets[1:]
    return solve_banded((1,1),ab,rhs,overwrite_ab=False,overwrite_b=False,check_finite=True)

def apply_measurements(operator,V,coefficients,splittings):
    """Apply freshly computed SOLA coefficient weights to a complete sweep."""
    split=operator.householder(np.asarray(splittings)*operator.op.inv_error)
    projected=V.T@split[1:]
    return coefficients.T@projected+operator.alpha*split[0]

def invert_from_basis(data,operator,U,V,alpha,beta,tradeoff=3.,width=.06,angular_width=.1,target_batch=128,kernel_indices=None):
    """Fit all 5,151 native targets in a freshly computed Lanczos subspace."""
    op=operator.op;nr,nt=op.nr,op.nt;k=len(alpha)
    widths=radial_target_widths(data,width=width)
    coeff=np.empty((k,op.np));transformed=np.empty((k+1,op.np))
    for lo in range(0,op.np,target_batch):
        idx=np.arange(lo,min(lo+target_batch,op.np))
        targets=gaussian_targets(data['mesh'],widths,idx,angular_width)
        rhs=op.sqrt_weight[:,None]*targets-operator.constraint_kernel[:,None]
        projected=U.T@rhs;transformed[:,idx]=projected
        coeff[:,idx]=solve_projected(alpha,beta,projected,tradeoff)
    rotation=apply_measurements(operator,V,coeff,data['splittings'])
    error=np.sqrt(np.sum(coeff**2,axis=0)+operator.alpha**2)
    if kernel_indices is None:kernel_indices=np.array([0,25,50,26*51+25,50*51+25,70*51,70*51+25,90*51+25,100*51+50])
    selected=coeff[:,kernel_indices]
    reduced=alpha[:,None]*selected
    lower=np.zeros((k+1,len(kernel_indices)));lower[:k]=reduced;lower[1:]+=beta[:,None]*selected
    kernels=(U@lower+operator.constraint_kernel[:,None])*op.inv_sqrt_weight[:,None]
    # Whitened estimator coefficient vectors for selected targets, used only
    # for uncertainty/correlation and exact normalization checks.
    reduced_coeff=V@selected
    full_coeff=np.vstack((np.zeros((1,len(kernel_indices))),reduced_coeff))
    full_coeff-=operator.tau*operator.v[:,None]*(operator.v@full_coeff)[None,:]
    full_coeff+=operator.q[:,None]
    covariance=full_coeff.T@full_coeff
    diagnostics=dict(formal_error=error.reshape(nr,nt),radial_widths=widths,kernel_indices=np.asarray(kernel_indices),averaging_kernels=kernels.T.reshape(-1,nr,nt),estimator_coefficients=full_coeff.T,covariance=covariance,coefficients=coeff,projected_targets=transformed,alpha=alpha,beta=beta,U=U,V=V)
    return {'reconstruction':rotation.reshape(1,nr,nt)},diagnostics

def reconstruct(data,iterations=1000,tradeoff=3.,width=.06,angular_width=.1,progress=False):
    """Complete native acquisition inversion; returns output and diagnostics."""
    op=RotationOperator(data);con=ConstrainedOperator(op)
    start=op.sqrt_weight*initial_target(data['mesh'])-con.constraint_kernel
    U,V,a,b=lanczos_bidiagonalize(con,start,iterations,progress)
    return invert_from_basis(data,con,U,V,a,b,tradeoff,width,angular_width)
