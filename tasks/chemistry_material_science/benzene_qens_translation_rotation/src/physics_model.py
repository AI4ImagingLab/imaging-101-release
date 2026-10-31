"""Classical symmetric l<=2 molecular kernel; physical meV coordinates."""
import numpy as np
from scipy.special import spherical_jn

LOWER=np.array([.01/1.5,.01/1.5,1e-5,.0225]+[.001]*12+[.0225]+[.001]*14)
UPPER=np.array([.2/1.5,1/1.5,.18,67.5]+[10.]*12+[67.5]+[10.]*14)

def lorentzian(energy,gamma):
    return gamma/(np.pi*(energy**2+gamma**2))

def kernel_basis(global_parameters,measurement):
    slope,rotation,u2=np.asarray(global_parameters,dtype=float)
    q=measurement['q'];e=measurement['energy'];n=e.size;delta=e[1]-e[0]
    omega=np.arange(-(n-1),n)*delta
    out=[]
    for j,Q in enumerate(q):
        translation=np.exp(-Q**2*u2/3)*lorentzian(omega,slope*Q**2)
        weights=np.array([(2*l+1)*spherical_jn(l,Q*2.48)**2 for l in range(3)])
        rotation_kernel=weights[1]*lorentzian(omega,rotation)+weights[2]*lorentzian(omega,3*rotation)
        rotation_kernel[n-1]+=weights[0]/delta
        molecular=np.convolve(rotation_kernel,translation,mode='same')
        out.append(np.convolve(measurement['resolution'][j],molecular,mode='valid'))
    return np.asarray(out)

def predict(parameters,data):
    p=np.asarray(parameters,dtype=float)
    return {name:kernel_basis(p[:3],data[name])/p[index]+p[start:stop,None] for name,index,start,stop in [('360',3,4,16),('197',16,17,31)]}

def weighted_sse(predictions,data):
    return float(sum(np.sum(((predictions[k]-data[k]['y'])/data[k]['sigma'])**2) for k in data))
