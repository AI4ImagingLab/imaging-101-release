import numpy as np
from scipy.optimize import least_squares
from .physics_model import gaussian_model
from .preprocessing import weights
KEYS=('log_amplitude','arrival_time','width','residual_dm','spectral_index','spectral_running')

def initial_parameters(data):
 profile=data['measurements'][data['good_channels']].mean(0);peak=int(np.argmax(profile));amplitude=max(profile[peak],1e-5)
 return np.array([np.log10(amplitude),data['times'][peak],(data['times'][1]-data['times'][0])/2,0.,0.,0.])

def fit_fixed(data,fixed=None,ff=8,ft=4,initial=None):
 fixed={} if fixed is None else fixed;w=weights(data);initial=initial_parameters(data) if initial is None else np.asarray(initial,dtype=float).copy()
 for name,value in fixed.items():initial[KEYS.index(name)]=value
 free=[i for i,name in enumerate(KEYS) if name not in fixed]
 def expand(values):
  q=initial.copy();q[free]=values;return q
 def residual(values):
  q=expand(values)
  if q[2]==0:return np.full(data['measurements'].size,1e100)
  model=gaussian_model(q,data['frequencies'],data['times'],ff=ff,ft=ft)
  return ((data['measurements']-model)*w[:,None]).ravel()
 result=least_squares(residual,initial[free],jac='2-point')
 q=expand(result.x);q[2]=abs(q[2])
 return {'reconstruction':q[None]},dict(optimizer_success=bool(result.success),cost=float(result.cost),nfev=int(result.nfev),message=result.message)

def reconstruct(data,ff=8,ft=4):return fit_fixed(data,ff=ff,ft=ft)
