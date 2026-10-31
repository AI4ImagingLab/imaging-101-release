"""MIT2025 Manuel Pagavino; local extraction of published classical baselines.

The source calls its second-order derivative penalty TGV. Its actual operator
is the vector (Dxx,Dyy), with pixelwise complex L2 coupling. This is not an
auxiliary-field Bredies TGV model nor the scalar summed Laplacian penalty.
"""
import numpy as np
import torch
import torch.nn.functional as F
from .physics_model import propagator,SOURCE_SHAPE
def difference(q,order=2):
    """Right-zero-padded forward finite differences; output(...,H,W,2), x then y."""
    if order not in (1,2):raise ValueError('order must be 1 or 2')
    shape=tuple(slice(n) for n in q.shape);p=F.pad(q,(0,order,0,order))
    parts=[]
    for axis in [-1,-2]:
        d=torch.diff(p,n=1,dim=axis) if order==1 else torch.roll(p,-2,axis)-2*torch.roll(p,-1,axis)+p
        parts.append(d[shape])
    return torch.stack(parts,-1)
def difference_adjoint(u,order=2):
    """Exact adjoint of the finite-ROI difference, including original corner handling."""
    p=F.pad(u,(0,0,order,0,order,0))
    if order==2:p[..., :2,:2,:]=p[..., -2:,-2:,:]
    result=0.
    for component,axis in [(0,-1),(1,-2)]:
        if order==1:
            shape=list(p.shape[:-1]);shape[axis]=1
            d=torch.diff(-p[...,component],dim=axis,prepend=torch.zeros(shape,dtype=p.dtype,device=p.device))
        else:d=torch.roll(p[...,component],2,axis)-2*torch.roll(p[...,component],1,axis)+p[...,component]
        result=result+d[...,order:,order:]
    return result
def data_gradient(q,p,G):
    """Gradient of ||Gq-p||_2^2 with respect to the two real components of q."""
    return 2*(G.mH@(G@q.reshape(*q.shape[:2],-1,1)-p.reshape(*p.shape[:2],-1,1))).reshape(q.shape)
def project_dual(u,weight):
    """Pixelwise complex-vector L2 ball of radius weight."""
    return u/torch.clamp(torch.linalg.norm(u,2,-1,keepdim=True)/weight,min=1)
def solve(pressure,method='TGV',iterations=20000,snr_db=None,return_history=False):
    """Return(q, diagnostics) with q complex64(40,50), fixed author amplitude/phase.

    No learned checkpoint, target mask, source-position truth or domain package.
    LN is the original closed-form SVD Tikhonov estimate. TV/TGV use explicit
    Condat-Vu primal/dual steps with first/second-order zero-edge derivatives.
    """
    if method not in ['LN','TV','TGV']:raise ValueError('method must be LN,TV,TGV')
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1:raise ValueError('positive integer iterations required')
    if snr_db is None:snr_db=30. if method=='LN' else 35.
    if not np.isfinite(snr_db):raise ValueError('finite SNR tuning parameter required')
    G=torch.from_numpy(propagator()[None,None]);p=torch.from_numpy(np.asarray(pressure,dtype=np.complex64)[None,None])
    q=torch.zeros((1,1,*SOURCE_SHAPE),dtype=torch.complex64)
    smax=torch.linalg.norm(G[0,0],2);pnorm=torch.linalg.norm(p.flatten())
    weight=smax*10**(-snr_db/20)*pnorm/np.sqrt(np.prod(SOURCE_SHAPE))
    tau=.5/smax**2
    diagnostic={'lambdaa':float(weight),'smax':float(smax),'tau':float(tau),'history':{}}
    if method=='LN':
        U,s,Vh=torch.linalg.svd(G,full_matrices=False)
        inverse=Vh.mH@(torch.diag_embed((s/(s*s+weight)).cfloat())@U.mH)
        q=(inverse@p.reshape(1,1,-1,1)).reshape(q.shape)
    else:
        order=1 if method=='TV' else 2
        # Original source fixes its derivative spectral-norm estimate on I100.
        # General matrix norm/eigendecomposition is allowed; the derivative is local.
        eye=torch.eye(100);eps=torch.linalg.norm(difference_adjoint(difference(eye,order),order),ord=2)
        beta=2*smax*smax;sigma=(1/eps)*(1/tau-.5*beta)
        diagnostic['sigma']=float(sigma);u=torch.zeros((*q.shape,2))
        for iteration in range(iterations):
            previous=q
            q=q-tau*data_gradient(q,p,G)
            q=q-tau*difference_adjoint(u,order)
            u=project_dual(u+sigma*difference(2*q-previous,order),weight)
            # Original relaxation rho=1 and fused sparsity mu=0: identities.
            if return_history and iteration+1 in [1,2,10,100,101,1000,5000,20000]:
                diagnostic['history'][iteration+1]={'q':q.numpy().copy(),'u':u.numpy().copy()}
    return q[0,0].numpy(),diagnostic
