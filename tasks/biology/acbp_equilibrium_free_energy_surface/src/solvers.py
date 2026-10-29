"""Complete native observed-corner initialization and global nonlinear fit."""
import numpy as np
from scipy.optimize import leastsq
from .preprocessing import flatten_observations
from .physics_model import fluorescence
def initial_parameters(temperature,denaturant,intensity):
 out=np.zeros(18);out[14:]=[70.,300.,(-130+16*90)/1000*(8.314e-3/1.985e-3),(-580+36*90)/1000*(8.314e-3/1.985e-3)]
 low=int(np.argmin(denaturant));high=int(np.argmax(denaturant))
 # Native distinct nearest corner at one molar inward from each extreme.
 low_other=low;candidate=denaturant[low]+1.-.01
 while low_other==low:candidate+=.01;low_other=int(np.argmin(abs(denaturant-candidate)))
 high_other=high;candidate=denaturant[high]-1.+.01
 while high_other==high:candidate-=.01;high_other=int(np.argmin(abs(denaturant-candidate)))
 for c in range(2):
  y=intensity[c];T0=temperature[low,0];i0=int(np.argmin(abs(temperature[low_other]-T0)));i1=int(np.argmin(abs(temperature[low]-(T0+5.))))
  if i1==0:i1=1
  bFD=(y[low_other,i0]-y[low,0])/(denaturant[low_other]-denaturant[low]);bFT=(y[low,i1]-y[low,0])/(temperature[low,i1]-T0);aF=y[low,0]+bFT*(25.-T0)-bFD*denaturant[low]
  Tu=temperature[high,-1];ju=int(np.argmin(abs(temperature[high_other]-Tu)));bUD=(y[high,-1]-y[high_other,ju])/(denaturant[high]-denaturant[high_other])
  bUT2,bUT,aU=np.polyfit(temperature[high]-25.,y[high]-denaturant[high]*bUD,2)
  out[c*7:(c+1)*7]=[aF,bFD,bFT,aU,bUD,bUT,bUT2]
 return out
def reconstruct(temperature,denaturant,normalized_intensity,initial=None):
 t,d,c,y=flatten_observations(temperature,denaturant,normalized_intensity)
 p0=initial_parameters(temperature,denaturant,normalized_intensity) if initial is None else np.asarray(initial).copy()
 def residual(p):return fluorescence(p,t,d,c)-y
 p,cov,info,message,code=leastsq(residual,p0,full_output=True,ftol=1e-12,xtol=1e-12,gtol=1e-12,maxfev=5000)
 if code not in [1,2,3,4]:raise RuntimeError(message)
 return dict(parameters=p,initial_parameters=p0,prediction=fluorescence(p,t,d,c),residual=info['fvec'],objective=float(np.dot(info['fvec'],info['fvec'])),evaluations=int(info['nfev']),status=code)
