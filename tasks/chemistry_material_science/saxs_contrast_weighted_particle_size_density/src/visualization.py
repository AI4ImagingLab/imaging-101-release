"""Review metrics and source-convention size integrals."""
import numpy as np
def density_metrics(estimate,reference):
    a=np.ravel(estimate);b=np.ravel(reference)
    return dict(ncc=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def size_integrals(density,radius,widths):
    mass=np.asarray(density)*widths;total=float(np.sum(mass))
    return dict(bin_sum_contrast=total,bin_weighted_rms_radius=float(np.sqrt(np.sum(radius**2*mass)/total)),source_trapezoid_contrast=float(np.trapezoid(density,radius)))
