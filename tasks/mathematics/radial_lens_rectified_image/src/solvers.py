"""Original algebraic stationary-point selection via explicit Sylvester resultant.

Adapted from AMI Research Group (2010-2014), CC-BY-NC-SA3.0 Spain.
NumPy polynomial roots replace the source's generic polynomial root primitive;
the task-specific quartic, resultant and candidate selection remain explicit.
"""
from itertools import permutations
import numpy as np
from .preprocessing import normalize_points
from .physics_model import determinant_polynomial
def resultant(dx,dy):
    matrix=np.zeros((6,6,5))
    for i in range(4):
        for row in range(3):
            matrix[row,i+row]=dx[3-i]
            matrix[row+3,i+row]=dy[3-i]
    out=np.zeros(19)
    for perm in permutations(range(6)):
        v=np.array([1.])
        for row,col in enumerate(perm):
            if not np.any(matrix[row,col]):break
            v=np.convolve(v,matrix[row,col])[:19]
        else:
            sign=(-1)**sum(perm[i]>perm[j] for i in range(6) for j in range(i+1,6))
            out[:len(v)]+=sign*v
    return out
def algebraic_minimum(polynomial):
    p=polynomial.copy()
    sx=p[4,0]**.25;sy=p[0,4]**.25
    if min(sx,sy)<=0:raise ValueError('Degenerate quartic')
    p/=sx**np.arange(5)[:,None];p/=sy**np.arange(5)[None,:]
    dx=np.zeros_like(p);dy=np.zeros_like(p)
    dx[:4]=np.arange(1,5)[:,None]*p[1:]
    dy[:,:4]=np.arange(1,5)[None,:]*p[:,1:]
    resolvent=resultant(dx,dy)
    roots=np.polynomial.polynomial.polyroots(resolvent)
    candidates=[]
    for root in roots:
        if abs(root.imag)>1e-15:continue
        y=float(root.real)
        for derivative in (dx,dy):
            coefficients=np.array([np.polynomial.polynomial.polyval(y,row) for row in derivative])[:4]
            for x in np.polynomial.polynomial.polyroots(coefficients):
                if abs(x.imag)>1e-12:continue
                x=float(x.real);value=np.polynomial.polynomial.polyval2d(x,y,p)
                if np.isfinite(value):candidates.append((value,x,y))
    if not candidates:raise ValueError('No real algebraic candidates')
    energy,x,y=min(candidates)
    return np.array([x/sx,y/sy]),dict(energy=energy,resultant=resolvent,candidates=candidates)
def infer_distortion(points,line_ids):
    normalized,factor=normalize_points(points)
    pair,_=algebraic_minimum(determinant_polynomial(normalized,line_ids))
    return np.array([1.,0.,pair[0]/factor**2,0.,pair[1]/factor**4])
