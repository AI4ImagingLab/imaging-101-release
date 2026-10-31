import numpy as np
from scipy.optimize import minimize
from .physics_model import usage_penalties,objective_gradient

def solve_sample(observation,endmembers,ratios,weights,penalty):
 n=len(endmembers);m=len(ratios);x=np.r_[np.full(n,1/n),np.zeros(m)]
 # Scale reaction coordinates by100umol/kg; fraction coordinates stay dimensionless.
 scale=np.r_[np.ones(n),np.full(m,100.)]
 def fun(v):
  f,g=objective_gradient(v*scale,observation,endmembers,ratios,weights,penalty)
  return f/10000.,g*scale/10000.
 a=np.r_[np.ones(n),np.zeros(m)]
 result=minimize(fun,x/scale,jac=True,method='SLSQP',bounds=[(0.,1.)]*n+[(0.,None)]*m,constraints=[dict(type='eq',fun=lambda v:np.dot(a,v)-1.,jac=lambda v:a)],options=dict(ftol=1e-13,maxiter=1500))
 if not result.success:raise RuntimeError(result.message)
 return result.x*scale,result.nit

def reconstruct(data,penalty_enabled=True,reaction_enabled=True):
 end=data['endmembers'];ratios=data['reaction_ratios'] if reaction_enabled else np.zeros((0,6));w=data['tracer_weights'];p=usage_penalties(data['sample_coordinates']) if penalty_enabled else np.zeros((len(data['observations']),9))
 q=[];iterations=[]
 for y,pi in zip(data['observations'],p):
  value,it=solve_sample(y,end,ratios,w,pi);q.append(value);iterations.append(it)
 q=np.array(q)
 return q[:,:9][None],q[:,9:][None],np.array(iterations)
