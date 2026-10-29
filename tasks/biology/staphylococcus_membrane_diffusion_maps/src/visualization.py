"""Presentation and common numeric definitions; no backend side effects."""
import numpy as np

def vector_metrics(estimate,reference):
    estimate=np.asarray(estimate).ravel();reference=np.asarray(reference).ravel()
    scale=float(np.ptp(reference));norm=float(np.linalg.norm(reference));enorm=float(np.linalg.norm(estimate))
    if scale<=0 or norm<=0:raise ValueError('Degenerate metric reference')
    return {'ncc':float(np.dot(estimate,reference)/(enorm*norm)) if enorm else 0.,'nrmse':float(np.sqrt(np.mean((estimate-reference)**2))/scale)}

def normalized_residual(observed,predicted):
    return np.sqrt(np.mean(((np.asarray(predicted)-observed)/observed[:,0,None])**2))

def nonlinear_mse(observed,predicted):
    """Paper same-sign-run weighted residual, normalized by first fitted lag.

    Weight is number of other members in the current contiguous same-sign run.
    This is a diagnostic, never an output-dependent evaluation selection mask.
    """
    residual=(np.asarray(observed)-predicted)/predicted[0]
    signs=np.sign(residual); boundaries=np.r_[0,np.flatnonzero(signs[1:]!=signs[:-1])+1,len(signs)]
    weighted=np.zeros_like(residual)
    for start,stop in zip(boundaries[:-1],boundaries[1:]):weighted[start:stop]=(stop-start-1)*residual[start:stop]**2
    return float(np.mean(weighted))
