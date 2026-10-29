"""Lambertian factorization and explicit masked backward-gradient normal equations.

Adapted mathematical stages from Grosse et al. ICCV2009 author code, exact
source hashes and qualifications in test/validation.md. No upstream imports.
"""
import numpy as np
from scipy.sparse import coo_matrix
def compose(shading: np.ndarray,reflectance: np.ndarray) -> np.ndarray:
    """Elementwise I=S*R, relative detector units."""
    return np.asarray(shading)*np.asarray(reflectance)
def gradients(image: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Row/column backward differences, zero first row/column; trailing axes retained."""
    a=np.asarray(image,dtype=np.float64);gy=np.zeros_like(a);gx=np.zeros_like(a)
    gy[1:]=a[1:]-a[:-1];gx[:,1:]=a[:,1:]-a[:,:-1]
    return gy,gx
def normal_equations(ty: np.ndarray,tx: np.ndarray,mask: np.ndarray,wy=None,wx=None):
    """Return A=D'WD, b=D'Wt, row-major support numbering (zero outside).

    Only edges with BOTH endpoints in the public mask contribute. Weights
    multiply squared residuals, not gradients before squaring.
    """
    m=np.asarray(mask,dtype=bool)
    if m.ndim!=2 or not m.any() or ty.shape!=m.shape or tx.shape!=m.shape:raise ValueError('Invalid masked gradient arrays')
    wy=np.ones(m.shape) if wy is None else np.asarray(wy);wx=np.ones(m.shape) if wx is None else np.asarray(wx)
    if wy.shape!=m.shape or wx.shape!=m.shape or np.any(wy<=0) or np.any(wx<=0):raise ValueError('Weights must be positive with matching shape')
    numbers=np.zeros(m.shape,dtype=np.int64);numbers[m]=np.arange(m.sum())
    ey=m[1:]&m[:-1];ex=m[:,1:]&m[:,:-1]
    a=np.concatenate([numbers[:-1][ey],numbers[:,:-1][ex]]);b=np.concatenate([numbers[1:][ey],numbers[:,1:][ex]])
    w=np.concatenate([wy[1:][ey],wx[:,1:][ex]]);target=np.concatenate([ty[1:][ey],tx[:,1:][ex]])
    matrix=coo_matrix((np.concatenate([w,w,-w,-w]),(np.concatenate([a,b,a,b]),np.concatenate([a,b,b,a]))),shape=(int(m.sum()),int(m.sum()))).tocsr()
    rhs=np.bincount(b,weights=w*target,minlength=int(m.sum()))-np.bincount(a,weights=w*target,minlength=int(m.sum()))
    return matrix,rhs,numbers
