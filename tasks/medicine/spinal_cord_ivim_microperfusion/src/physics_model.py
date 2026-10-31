"""One-pool IVIM model; Dstar is extra pseudodiffusion, in mm²/s."""
import numpy as np

def signal(b,S0,D,f,Dstar):
    return S0*np.exp(-b*D)*(f*np.exp(-b*Dstar)+1-f)

def jacobian(b,S0,D,f,Dstar):
    slow=np.exp(-b*D);fast=np.exp(-b*Dstar);mixture=f*fast+1-f
    return np.stack([mixture*slow,-b*S0*mixture*slow,S0*(fast-1)*slow,-b*S0*f*fast*slow],axis=-1)

def log_signal(b,lnS0,D):
    return lnS0-b*D
