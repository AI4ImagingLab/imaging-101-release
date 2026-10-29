"""Local dilute-sphere Fredholm operator. Original MIT notice: see LICENSE."""
import numpy as np
def sphere_amplitude(qr):
    x=np.asarray(qr,dtype=float);out=np.empty_like(x);small=x<1e-3
    x2=x[small]**2;out[small]=1-x2/10+x2**2/280
    y=x[~small];out[~small]=3*(np.sin(y)-y*np.cos(y))/y**3
    return out
def build_kernel(q,radius):
    x=np.asarray(q)[:,None]*np.asarray(radius)[None,:]
    volume=(4./3.)*np.pi*np.asarray(radius)**3
    return volume[None,:]*sphere_amplitude(x)**2*1e-4
def forward(density,background,q,radius,widths):
    return build_kernel(q,radius)@(np.asarray(density)*widths)+float(background)
def weighted_objective(bin_mass,kernel,target,error):
    residual=(kernel@bin_mass-target)/error
    return float(residual@residual)
