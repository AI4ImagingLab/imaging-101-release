"""Local Gaussian/RPL/cold-plasma acquisition mathematics; source license in NOTICE."""
import numpy as np
from scipy.special import erf
K=4149.377593360996;NOMINAL_DM=623.6785965856119;REF_FREQ=400.1953125

def gaussian_model(q,frequencies,times,ff=8,ft=4,analytic_time=False):
 logamp,t0,width,dm,index,running=q;df=frequencies[1]-frequencies[0];dt=times[1]-times[0];out=np.empty((len(frequencies),len(times)))
 ts=np.linspace(times[0]-dt/2+dt/(2*ft),times[-1]+dt/2-dt/(2*ft),len(times)*ft)
 for lo in range(0,len(frequencies),128):
  fc=frequencies[lo:lo+128];sub=np.linspace(fc-df/2+df/(2*ff),fc+df/2-df/(2*ff),ff,axis=-1)
  arrival=t0+(NOMINAL_DM+dm)*K*(sub**-2-REF_FREQ**-2)-NOMINAL_DM*K*(fc[:,None]**-2-REF_FREQ**-2)
  lognu=np.log(sub/REF_FREQ);spectral=np.exp(index*lognu+running*lognu**2)
  if analytic_time:
   upper=(times[None,None,:]+dt/2-arrival[:,:,None])/(np.sqrt(2)*width);lower=upper-dt/(np.sqrt(2)*width)
   p=np.sqrt(np.pi/2)*width/dt*(erf(upper)-erf(lower));result=(p*spectral[:,:,None]).mean(1)
  else:
   p=np.exp(-.5*((ts[None,None,:]-arrival[:,:,None])/width)**2)*spectral[:,:,None]
   result=p.mean(1).reshape(len(fc),len(times),ft).mean(2)
  out[lo:lo+len(fc)]=10**logamp*result
 return out
