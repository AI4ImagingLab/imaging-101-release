"""Explicit original single-precision CG updates. GPL-3.0-or-later; Triad 2024."""
import numpy as np
from .physics_model import matrix_apply, build_rhs, spacing
from .preprocessing import PRESSURE_SCALE_PA
RELATIVE_TOLERANCE=1e-4
ABSOLUTE_TOLERANCE=1e-6

def sequential_dot(a,b):
    """Match upstream CPU single-thread float32 reduction order."""
    return np.cumsum((a*b).ravel(),dtype=np.float32)[-1]

def solve(gradient,coordinate,relative_tolerance=RELATIVE_TOLERANCE,absolute_tolerance=ABSOLUTE_TOLERANCE):
    """Return physical pressure float64(H,W), and CG history float64(K,3)."""
    dx=spacing(coordinate);rhs=build_rhs(gradient,dx)
    p=np.zeros_like(rhs);r=rhs.copy();direction=r.copy()
    initial=float(np.sqrt(float(sequential_dot(r,r))));history=[]
    if initial==0:return p.astype(np.float64),np.empty((0,3),np.float64)
    for k in range(p.size):
        rr=sequential_dot(r,r);ap=matrix_apply(direction,dx)
        denominator=sequential_dot(direction,ap)
        if not np.isfinite(denominator) or denominator<=0:raise ArithmeticError("CG lost positive curvature")
        alpha=np.float32(rr/denominator)
        new_r=r-alpha*ap;p+=alpha*direction
        norm=float(np.sqrt(float(sequential_dot(new_r,new_r))))
        history.append((k,norm/initial,norm))
        if norm/initial<=relative_tolerance or norm<=absolute_tolerance:
            break
        if norm/initial>1e3:raise ArithmeticError("CG diverged")
        beta=np.float32(sequential_dot(new_r,new_r)/rr)
        direction=beta*direction+new_r;r=new_r
    else:raise ArithmeticError("CG iteration limit")
    # Author display gauge: MATLAB pixel (2,2), zero-based index (1,1).
    result=(p.astype(np.float64)-float(p[1,1]))*PRESSURE_SCALE_PA
    return result,np.asarray(history,np.float64)
