"""Orthographic distant-light Lambertian model plus additive corruptions."""
import numpy as np
def forward(coefficients,lights):
    """Albedo-weighted normals (...,3), illumination (L,3) -> (...,L)."""
    return np.asarray(coefficients)@np.asarray(lights).T
def adjoint(measurements,lights):
    """Euclidean adjoint of the linear light operator; no clamping."""
    return np.asarray(measurements)@np.asarray(lights)
def normalize_rows(coefficients):
    """Unit normal rows, retaining zero rows as zero (background)."""
    c=np.asarray(coefficients,dtype=np.float64);norm=np.linalg.norm(c,axis=1,keepdims=True)
    # sklearn original leaves near-zero vectors (<10 machine eps) unscaled.
    norm[norm<10*np.finfo(c.dtype).eps]=1.
    return c/norm
def angular_errors(estimate,truth,support):
    """Degrees at every known foreground pixel; no sign/phase alignment."""
    a=np.asarray(estimate)[support];b=np.asarray(truth)[support]
    norms=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    if np.any(norms<=0):raise ValueError('Undefined normal orientation')
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/norms,-1,1)))
