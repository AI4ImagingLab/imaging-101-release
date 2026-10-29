"""Local rank3 factorization and complete selected rigid helper."""
import numpy as np
from .preprocessing import centre_and_scale
from .physics_model import metric_upgrade

def reconstruct(tracks):
    """Return shape[P,3]pixels,cameras[F,2,3],translations[F,2],diagnostics."""
    W,T,scale=centre_and_scale(tracks)
    U,s,Vh=np.linalg.svd(W,full_matrices=False)
    motion=U[:,:3]*np.sqrt(s[:3]);shape=np.sqrt(s[:3])[:,None]*Vh[:3]
    M,S,info=metric_upgrade(motion,shape)
    info.update(normalized=W,translation=T,scale=scale,singular_values=s,motion_affine=motion,shape_affine=shape)
    return (scale*S).T,M.reshape(len(tracks),2,3),T.reshape(len(tracks),2),info
