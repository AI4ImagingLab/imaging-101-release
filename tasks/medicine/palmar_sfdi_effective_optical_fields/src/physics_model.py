import numpy as np
REFRACTIVE_INDEX=1.4

def diffuse_reflectance(parameters,frequencies,index=REFRACTIVE_INDEX):
    mua,mus=np.asarray(parameters,dtype=float);f=np.asarray(frequencies,dtype=float);transport=mua+mus
    reflected=-1.440/(index**2)+.710/index+.668+.0636*index;a=(1-reflected)/(2*(1+reflected))
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        attenuation=np.sqrt(3*mua*transport+(2*np.pi*f)**2);ratio=attenuation/transport
        value=3*a*(mus/transport)/(ratio+1)/(ratio+3*a)
    return value

def objective(parameters,frequencies,observed,index=REFRACTIVE_INDEX):
    r=diffuse_reflectance(parameters,frequencies,index)-observed
    return float(np.sum(r*r))
