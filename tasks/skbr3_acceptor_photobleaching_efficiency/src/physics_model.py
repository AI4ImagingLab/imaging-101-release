"""Explicit corrected donor-quenching acquisition and its closed inverse.

Formula extracted from AccPbFRET, Roszik/Szollosi/Vereb GPL-3.0-or-later.
"""
import numpy as np

def donor_acquisition(efficiency,donor_brightness,acceptor_brightness,factors):
    """Return Dpre,Dpost,Apre,Apost (ADU) from dimensionless E,gamma,delta,epsilon,alpha."""
    gamma,delta,epsilon,alpha=np.asarray(factors,dtype=np.float64)
    e=np.asarray(efficiency,dtype=np.float64);d0=np.asarray(donor_brightness,dtype=np.float64);a=np.asarray(acceptor_brightness,dtype=np.float64)
    return (d0*(1-e)+delta*a,d0*(1-alpha*e)/gamma+(alpha*delta+(1-alpha)*epsilon)*a,a,alpha*a)

def corrected_efficiency(donor_before,donor_after,acceptor_before,factors):
    """Preserve the upstream float32/double expression boundaries, no clipping."""
    gamma,delta,epsilon,alpha=np.asarray(factors,dtype=np.float32)
    db=np.asarray(donor_before,dtype=np.float32);da=np.asarray(donor_after,dtype=np.float32);ab=np.asarray(acceptor_before,dtype=np.float32)
    x=db-ab*delta
    a,g,d,e=map(float,(alpha,gamma,delta,epsilon))
    y=np.asarray(g*(da.astype(np.float64)-(a*d+e*(1-a))*ab.astype(np.float64))-a*x.astype(np.float64),dtype=np.float32)
    return np.asarray(1-(1-a)*x.astype(np.float64)/y.astype(np.float64),dtype=np.float32)
