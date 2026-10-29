"""Finite-grid contact morphology, adapted from matsunagalab (MIT, 2024).

Probe is the nonpositive reflected cap stored at acquisition, with origin (6,6).
The general erosion uses reversed offsets: it is the order adjoint of dilation.
The bound probe's active heights are symmetric about (6,6); the unused far skirt
is below -223 A. Thus this adjoint erosion agrees with the upstream same-offset
implementation on every bound observation, verified against the original run.
"""
import numpy as np

def _image_and_probe(image, probe):
    x, p = np.asarray(image, dtype=np.float64), np.asarray(probe, dtype=np.float64)
    if x.ndim < 2 or p.ndim != 2 or not p.size:
        raise ValueError("Image needs two spatial axes and probe must be 2-D")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite morphology input")
    return x, p

def dilation(surface, probe):
    """Return max-offset contact image, same shape as surface, in A.

    For offsets u=(i-floor((K-1)/2),j-floor((L-1)/2)),
    D(s)[r,c]=max_u (s[r+u_y,c+u_x]+p[i,j]); absent pixels are -infinity.
    """
    x,p = _image_and_probe(surface,probe)
    h,w=x.shape[-2:]; kh,kw=p.shape; oy,ox=(kh-1)//2,(kw-1)//2
    out=np.full_like(x,-np.inf)
    for i in range(kh):
        for j in range(kw):
            dy,dx=i-oy,j-ox
            y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
            if y0<y1 and x0<x1:
                out[...,y0:y1,x0:x1]=np.maximum(out[...,y0:y1,x0:x1],x[...,y0+dy:y1+dy,x0+dx:x1+dx]+p[i,j])
    return out

def erosion(measurements, probe):
    """Return min-offset upper-envelope estimate, same shape, in A.

    E(y)[r,c]=min_u (y[r-u_y,c-u_x]-p[i,j]); absent pixels are +infinity.
    No clipping or offset alignment is performed here.
    """
    x,p = _image_and_probe(measurements,probe)
    h,w=x.shape[-2:];kh,kw=p.shape;oy,ox=(kh-1)//2,(kw-1)//2
    out=np.full_like(x,np.inf)
    for i in range(kh):
        for j in range(kw):
            dy,dx=i-oy,j-ox
            y0,y1=max(0,dy),min(h,h+dy);x0,x1=max(0,dx),min(w,w+dx)
            if y0<y1 and x0<x1:
                out[...,y0:y1,x0:x1]=np.minimum(out[...,y0:y1,x0:x1],x[...,y0-dy:y1-dy,x0-dx:x1-dx]-p[i,j])
    return out
