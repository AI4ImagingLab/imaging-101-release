"""Explicit contact-distance model, Poisson likelihood and MDS2 derivatives.

Adapted from Pastis, BSD-3-Clause, Nelle Varoquaux and contributors.
Only positive observed pairs enter the selected original conditional objective.
"""
import numpy as np

def wish_distances(row,col,counts,bias,alpha=-3.,beta=1.):
    """Contact-to-distance power law; distance units are fixed by beta=1."""
    if alpha>=0 or beta<=0:raise ValueError('alpha must be negative and beta positive')
    c=np.asarray(counts,dtype=np.float64).copy()
    b=np.asarray(bias).reshape(-1)
    c/=b[row]*b[col]
    wish=c/beta
    wish[wish!=0]**=1./alpha
    return wish

def pair_distances(coordinates,row,col):
    x=np.asarray(coordinates,dtype=np.float64).reshape(-1,3)
    return np.sqrt(((x[row]-x[col])**2).sum(axis=1))

def intensity(coordinates,row,col,bias,alpha=-3.,beta=1.):
    b=np.asarray(bias).reshape(-1)
    return b[row]*b[col]*beta*pair_distances(coordinates,row,col)**alpha

def poisson_objective(coordinates,row,col,counts,bias,alpha=-3.,beta=1.):
    """Sum(lambda-c*log(lambda)); omitted factorials are coordinate-independent."""
    lam=intensity(coordinates,row,col,bias,alpha,beta)
    value=lam.sum()-(counts*np.log(lam)).sum()
    if not np.isfinite(value):raise ValueError('Nonfinite Poisson objective')
    return float(value)

def poisson_gradient(coordinates,row,col,counts,bias,alpha=-3.,beta=1.):
    x=np.asarray(coordinates,dtype=np.float64).reshape(-1,3)
    distance=pair_distances(x,row,col)
    lam=intensity(x,row,col,bias,alpha,beta)
    difference=x[row]-x[col]
    edge=-((counts/lam-1)*lam*alpha/(distance**2))[:,None]*difference
    gradient=np.zeros(x.shape)
    # Keep source per-locus reduction order: nonconvex optimization is sensitive
    # to changing a sequential sum into an atomic/scatter accumulation.
    for k in range(x.shape[0]):
        gradient[k]+=edge[row==k].sum(axis=0)
        gradient[k]-=edge[col==k].sum(axis=0)
    return gradient

def mds_objective(coordinates,row,col,wish):
    distance=pair_distances(coordinates,row,col)
    return float(((distance-wish)**2/wish**2).sum())

def mds_gradient(coordinates,row,col,wish):
    x=np.asarray(coordinates,dtype=np.float64).reshape(-1,3)
    distance=pair_distances(x,row,col)
    edge=2*((distance-wish)/distance/wish**2)[:,None]*(x[row]-x[col])
    gradient=np.zeros(x.shape)
    for k in range(x.shape[0]):
        gradient[k]+=edge[row==k].sum(axis=0)
        gradient[k]-=edge[col==k].sum(axis=0)
    return gradient

def alpha_profile(coordinates,row,col,counts,bias,alpha):
    """PM2 scalar exponent objective after analytically profiling positive beta."""
    d=pair_distances(coordinates,row,col);b=np.asarray(bias).ravel()
    beta=counts.sum()/((d**alpha)*b[row]*b[col]).sum()
    lam=beta*d**alpha*b[row]*b[col]
    loglike=counts*np.log(beta)+alpha*counts*np.log(d)+counts*np.log(b[row]*b[col])
    loglike-=lam
    gradient=-beta*(b[row]*b[col]*d**alpha*np.log(d)).sum()+(counts*np.log(d)).sum()
    return float(-loglike.sum()),float(-gradient),float(beta)

def all_distances(coordinates,support):
    """All unique supported-locus distances for scientific geometry comparison."""
    x=np.asarray(coordinates)[np.asarray(support,dtype=bool)];i,j=np.triu_indices(len(x),1)
    return np.sqrt(((x[i]-x[j])**2).sum(1))

def align_for_display(coordinates,reference,support,allow_scale=False):
    """Explicit O(3) alignment for review only; the Judge never invokes it.

    Scale alignment is reserved for comparing different genomic resolutions or
    different fitted contact-law scales, and is labeled by each notebook panel.
    """
    p=np.asarray(coordinates,dtype=float);r=np.asarray(reference,dtype=float);m=np.asarray(support,dtype=bool)
    pc=p[m]-p[m].mean(0);rc=r[m]-r[m].mean(0);u,s,vt=np.linalg.svd(pc.T@rc);rotation=u@vt
    scale=float(s.sum()/np.sum(pc*pc)) if allow_scale else 1.
    out=np.zeros_like(p);out[m]=scale*pc@rotation+r[m].mean(0)
    return out,scale
