"""Generate only explicitly labeled maintainer read-retention sensitivities.
The bound measurements are the original real counts, never generated here.
A binomial draw thins each count independently at retention probability p,
seeded by NumPy RandomState; units are sequencing counts. No clipping,
interpolation, output-informed support, or hidden cell composition is used.
"""
import numpy as np

def thin_counts(counts,retention,seed):
    x=np.asarray(counts)
    if x.dtype.kind not in "iu" or np.any(x<0):raise ValueError("Nonnegative integer counts required")
    if not 0<retention<=1:raise ValueError("Retention must lie in (0,1]")
    return np.random.RandomState(seed).binomial(x,retention).astype(x.dtype)
