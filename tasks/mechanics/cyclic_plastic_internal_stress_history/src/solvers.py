import numpy as np
from scipy.optimize import fmin
from .physics_model import internal_states

def initial_parameters(rng):
 total=rng.uniform(100,400);a=rng.uniform(size=4);a/=a.sum();a=total*a
 while np.any(a>200):
  a=rng.uniform(size=4);a/=a.sum();a=total*a
 bar=rng.uniform(0,500);base=[rng.uniform(15,250),rng.uniform(100,10000),1/rng.uniform(1/.033333,1/.00666666)]
 c=np.array([np.exp(rng.uniform(np.log(lo),np.log(hi))) for lo,hi in [(1000,50000),(10,5000),(10,2000),(10,2000)]])
 ids=np.argsort(c[:3])[::-1];return np.r_[base,c[ids],c[3],a[ids],a[3],bar]

def fit_parameters(plastic_strain,stress,starts=16,seed=0):
 rng=np.random.RandomState(seed);records=[];best=None
 def loss(q):
  if np.any(q<1e-5) or not np.isfinite(q).all():return np.inf
  pred=internal_states(q,plastic_strain)[2];return float(np.mean((pred-stress)**2))
 for i in range(starts):
  q0=initial_parameters(rng);q,v,nit,ncall,status=fmin(loss,q0,disp=False,full_output=True,maxiter=12000,maxfun=24000,xtol=1e-8,ftol=1e-8)
  records.append(dict(start=i,mse=float(v),nit=nit,ncall=ncall,status=int(status),parameters=q.tolist()))
  if best is None or v<best[0]:best=(float(v),q)
 return best[1],records

def reconstruct(data,starts=16,seed=0):
 q,records=fit_parameters(data['plastic_strain'],data['measured_stress'],starts=starts,seed=seed);alpha,kiso,stress=internal_states(q,data['plastic_strain'])
 return dict(parameters=q[None],reconstruction=alpha[None],isotropic_stress=kiso[None]),records


