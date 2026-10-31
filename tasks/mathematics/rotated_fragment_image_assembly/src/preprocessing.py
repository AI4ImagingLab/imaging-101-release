"""Explicit source MGC edge statistics and bidirectional strip compatibility.

Independent mathematical implementation of Gallagher/Huroyan source conventions.
All public tile values remain original uint16 digital numbers, not normalized.
"""
import numpy as np


def load_data(path):
    """Load public tiles, returning original uint16 (N,P,P,3)."""
    with np.load(path, allow_pickle=False) as data:
        tiles = data['tiles']
        if tiles.dtype != np.uint16 or tiles.ndim != 5 or tiles.shape[0] != 1:
            raise ValueError('tiles must have batch-first uint16 shape (1,N,P,P,3)')
        tiles = tiles[0].copy()
    if tiles.shape[1] != tiles.shape[2] or tiles.shape[-1] != 3:
        raise ValueError('square RGB tiles required')
    return tiles


def compute_attributes(tiles):
    """Return four top/right/bottom/left source-compatible edge dictionaries.

    pix/pixR (N,3P), D_mu (N,3), D_cov (3,3,N) are float64.
    The RGB channel gradients and means are evaluated in float32 first.
    """
    x = np.asarray(tiles, dtype=np.float32)
    if x.ndim != 4 or x.shape[1] != x.shape[2] or x.shape[-1] != 3:
        raise ValueError('expected (N,P,P,3) square RGB tiles')
    dummy = np.array([[0,0,0],[1,1,1],[-1,-1,-1],[0,0,1],[0,1,0],[1,0,0],[-1,0,0],[0,-1,0],[0,0,-1]], np.float32)
    strips = [x[:,0], x[:,:,-1], x[:,-1,::-1], x[:,::-1,0]]
    grads = [x[:,0]-x[:,1], x[:,:,-1]-x[:,:,-2], x[:,-1]-x[:,-2], x[:,:,0]-x[:,:,1]]
    result=[]
    for strip, gradient in zip(strips, grads):
        # Source preallocated double arrays receive flattened single strips.
        pix=strip.transpose(0,2,1).reshape(len(x),-1).astype(np.float64)
        pixr=strip[:,::-1].transpose(0,2,1).reshape(len(x),-1).astype(np.float64)
        # Octave/MATLAB mean(single matrix) uses single arithmetic.
        mean=np.mean(gradient,axis=1,dtype=np.float32).astype(np.float64)
        samples=np.concatenate((gradient,np.broadcast_to(dummy,(len(x),9,3))),axis=1).astype(np.float64)
        centered=samples-np.mean(samples,axis=1,keepdims=True)
        cov=np.einsum('npi,npj->nij',centered,centered)/(samples.shape[1]-1)
        result.append({'pix':pix,'pixR':pixr,'D_mu':mean,'D_cov':cov.transpose(1,2,0)})
    return result


def mahalanobis_squared_spd(covariance, rgb_vectors):
    """Evaluate x.T C^-1 x via explicit 3x3 SPD Cholesky whitening.

    C has shape (...,3,3), x (...,P,3), all arithmetic float64. Covariance
    must be finite, exactly symmetric and positive definite. No clipping,
    diagonal jitter, target-dependent values or data-specific branches occur.
    """
    if np.iscomplexobj(covariance) or np.iscomplexobj(rgb_vectors):
        raise ValueError('real covariance and RGB vectors required')
    c=np.asarray(covariance,dtype=np.float64)
    x=np.asarray(rgb_vectors,dtype=np.float64)
    if c.ndim<2 or c.shape[-2:]!=(3,3) or x.ndim<2 or x.shape[-1]!=3:
        raise ValueError('expected covariance (...,3,3) and RGB vectors (...,P,3)')
    np.broadcast_shapes(c.shape[:-2],x.shape[:-2])
    if not np.isfinite(c).all() or not np.isfinite(x).all():
        raise ArithmeticError('nonfinite covariance or RGB vector')
    if np.any(c!=np.swapaxes(c,-1,-2)):
        raise ArithmeticError('covariance is not symmetric')
    p00=c[...,0,0]
    if np.any(p00<=0):raise ArithmeticError('nonpositive first Cholesky pivot')
    with np.errstate(divide='raise',invalid='raise',over='raise'):
        l00=np.sqrt(p00)
        l10=c[...,1,0]/l00
        l20=c[...,2,0]/l00
        p11=c[...,1,1]-l10*l10
        if not np.isfinite(p11).all() or np.any(p11<=0):
            raise ArithmeticError('nonpositive or nonfinite second Cholesky pivot')
        l11=np.sqrt(p11)
        l21=(c[...,2,1]-l20*l10)/l11
        p22=c[...,2,2]-l20*l20-l21*l21
        if not np.isfinite(p22).all() or np.any(p22<=0):
            raise ArithmeticError('nonpositive or nonfinite third Cholesky pivot')
        l22=np.sqrt(p22)
        z0=x[...,0]/l00[...,None]
        z1=(x[...,1]-l10[...,None]*z0)/l11[...,None]
        z2=(x[...,2]-l20[...,None]*z0-l21[...,None]*z1)/l22[...,None]
        quadratic=z0*z0+z1*z1+z2*z2
    if not np.isfinite(quadratic).all() or np.any(quadratic<0):
        raise ArithmeticError('invalid whitened covariance quadratic form')
    return quadratic


def compute_mgc(tiles, attributes=None):
    """Compute all ordered pairs and 16 orientation channels, float32 (N,N,16).

    MGC is sum of per-pixel RGB Mahalanobis lengths in both directions.
    Explicit 3x3 Cholesky whitening evaluates the same SPD quadratic forms.
    """
    a=compute_attributes(tiles) if attributes is None else attributes
    n=len(tiles);p=np.asarray(tiles).shape[1]
    opp=np.array([[3,4,1,2],[4,1,2,3],[1,2,3,4],[2,3,4,1]])-1
    mirror=np.array([3,4,1,2,16,13,14,15,9,10,11,12,6,7,8,5])-1
    score=np.full((n,n,16),np.inf,np.float32)
    for edge in range(4):
        for r in range(4):
            s=opp[r,edge];layer=edge+4*r
            for i in range(n-1):
                x=(a[edge]['pix'][i]-a[s]['pixR'][i+1:]).reshape(-1,3,p).transpose(0,2,1)
                u=x-a[s]['D_mu'][i+1:,None,:]
                v=-x-a[edge]['D_mu'][i,None,:]
                cov=a[s]['D_cov'][:,:,i+1:].transpose(2,0,1)
                qu=mahalanobis_squared_spd(cov,u)
                qv=mahalanobis_squared_spd(a[edge]['D_cov'][:,:,i],v)
                if np.any(qu<0) or np.any(qv<0):
                    raise ArithmeticError('negative covariance quadratic form')
                d=np.sum(np.sqrt(qu)+np.sqrt(qv),axis=1).astype(np.float32)
                score[i,i+1:,layer]=d
                score[i+1:,i,mirror[layer]]=d
    return score
