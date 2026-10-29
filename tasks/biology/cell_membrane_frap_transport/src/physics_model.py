"""Explicit compressed heat Green-function Gram operator (HiFRAP)."""
import numpy as np
from scipy.linalg.lapack import dpotrf,dtrtri
from scipy.sparse.linalg import svds

def compression_matrix(nx,nq=9):
    """Orthonormal real Fourier rows: DC, ascending cosine, descending sine."""
    q=np.arange(1,nq//2+1)/nx*np.pi*2
    x=np.arange(nx)
    return np.vstack((np.ones(nx)/np.sqrt(2),np.cos(q[:,None]*x),np.sin(q[::-1,None]*x)))*np.sqrt(2)/np.sqrt(nx)

def heat_gram(diffusion,epsilon,nt=41,nx=89,nq=9,beta=0.):
    """Return (nt*nq,nt*nq) Gram matrix; time/frame and length/pixel units.

    Infinite-line heat kernel with PSF scale one pixel, sampled on the finite
    observation window. This is not periodic diffusion on that window.
    """
    x=np.arange(nx); t=np.arange(2*nt)
    xx,tt=np.meshgrid(x,t)
    width=diffusion*tt+1.
    kernel=np.exp(-xx**2/(4*width))/np.sqrt(4*np.pi*width)*np.exp(-(beta+epsilon)*tt)
    c=compression_matrix(nx,nq)
    distance=np.abs(x[:,None]-x[None,:])
    blocks=[c@kernel[i,distance]@c.T for i in range(2*nt-1)]
    gram=np.empty((nt,nq,nt,nq))
    for i in range(nt):
        for j in range(nt): gram[i,:,j,:]=blocks[i+j]
    return gram.reshape(nt*nq,nt*nq)

def profile_residual(diffusion,compressed,epsilon,background,stationary=0.,model='diffusion',beta=0.,return_all=False):
    """Marginalize the unknown initial profile using original ridge projector.

    Normalized residual = lambda*(K+lambda I)^-1*(y-h)/sqrt(trace(P²)).
    API-seeded ARPACK fixes execution convention without changing the ridge.
    """
    nt,nq=compressed.shape; nx=89
    gram=heat_gram(float(np.asarray(diffusion).reshape(-1)[0]),epsilon,nt,nx,nq,beta)
    maximum=svds(gram,k=1,return_singular_vectors=False,rng=np.random.default_rng(1729))
    ridge=maximum*(nt*nq*np.finfo(gram.dtype).eps)
    factor,info=dpotrf(gram+ridge*np.eye(len(gram)),lower=True)
    if info: raise ValueError(f'Gram Cholesky failure {info}')
    inverse_factor,info=dtrtri(factor,lower=True,unitdiag=False)
    if info: raise ValueError(f'Triangular inverse failure {info}')
    inverse=inverse_factor.T@inverse_factor
    trace=np.squeeze(ridge**2*np.sum(inverse**2))
    shifted=compressed.copy()
    shifted[:,0]-=background*np.sqrt(nx)
    if model!='diffusion': shifted[:,0]-=stationary*np.exp(-epsilon*np.arange(nt))*np.sqrt(nx)
    residual=(ridge*inverse@shifted.ravel()).reshape(nt,nq)/np.sqrt(trace)
    if return_all: return residual,float(np.sum(residual**2)),float(trace)
    return residual.ravel()
