"""Metric definitions and physical-population review helpers."""
import numpy as np
def metrics(estimate,reference):
    a=np.ravel(estimate);b=np.ravel(reference);norm=np.linalg.norm(a)*np.linalg.norm(b)
    return dict(ncc=float(a@b/norm) if norm>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def normalized_density(axis,density):
    area=float(np.trapezoid(density,axis))
    if area<=0:raise ValueError('Cannotnormalize empty porepopulation')
    return density/area
