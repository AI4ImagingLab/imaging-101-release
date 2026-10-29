"""Explicit four-parameter fluorescence response and its published inverse.

Adapted from Ratio2pH (Roessling et al.), licensed Zenodo v3.0 CC-BY-4.0
release, byte-identical author source revision identified in NOTICE.
"""
import numpy as np

TOP=1.1171
BOTTOM=-0.0295
HILL_SLOPE=1.2133
LOG_IC50=6.2110
EPS=0.0001


def normalize_ratio(ratio,lower,upper):
    """Dimensionless experiment-specific two-reference normalization."""
    if not np.isfinite([lower,upper]).all() or upper<=lower:raise ValueError('Need ordered finite calibration levels')
    return (np.asarray(ratio,dtype=np.float64)-lower)/(upper-lower)


def fluorescence_response(ph,top=TOP,bottom=BOTTOM,hill_slope=HILL_SLOPE,log_ic50=LOG_IC50):
    """Forward normalized HPTS response; pH and output are dimensionless."""
    if top<=bottom or abs(hill_slope)<1e-12:raise ValueError('Invalid sigmoid parameters')
    return bottom+(top-bottom)/(1.0+np.power(10.0,hill_slope*(log_ic50-np.asarray(ph,dtype=np.float64))))


def invert_response(normalized,top=TOP,bottom=BOTTOM,hill_slope=HILL_SLOPE,log_ic50=LOG_IC50,eps=EPS):
    """Return float64 pH and original asymptote-clipping indicator.

    The epsilon clipping is part of the original inverse. It is not a
    restriction to pH 5..7, which was only the source display LUT range.
    """
    if top<=bottom or abs(hill_slope)<1e-12 or not 0<eps<(top-bottom)/2:raise ValueError('Invalid inverse sigmoid parameters')
    normalized=np.asarray(normalized,dtype=np.float64)
    clipped=np.clip(normalized,bottom+eps,top-eps)
    val=(top-bottom)/(clipped-bottom)-1.0
    ph=log_ic50-np.log(val)/(hill_slope*np.log(10.0))
    return ph,(normalized<bottom+eps)|(normalized>top-eps)
