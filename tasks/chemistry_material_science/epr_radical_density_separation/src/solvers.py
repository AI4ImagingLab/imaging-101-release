"""Original CP2016 EPR updates extracted explicitly; MIT(c)2024 Remy Abergel."""
import math
import numpy as np
from .physics_model import grad2d,div2d,NormalOperator

def squared_norm_sequence(u):return sum(float(np.sum(x*x)) for x in u)

def lipschitz_power_iteration(init,normal,tol=1e-5,nitermax=1000):
 x=np.array(init,copy=True);value=math.sqrt(squared_norm_sequence(x));previous=value*(2+2*tol);iteration=1
 while abs(value-previous)>abs(value)*tol and iteration<nitermax:
  x=normal(x);previous=value;value=math.sqrt(squared_norm_sequence(x));x=x/value;iteration+=1
 return 1.05*math.sqrt(squared_norm_sequence(normal(x))/squared_norm_sequence(x))

def primal_dual(init,normal,backprojection,Lf,weight,tol=1e-5,nitermax=5000,positivity=False,energy=None):
 u=[np.array(x,copy=True) for x in init];ubar=[np.array(x,copy=True) for x in init];p=[np.zeros((2,)+x.shape) for x in u]
 tau=.5/Lf;sigma=Lf/((math.sqrt(8)*weight)**2)
 g=normal(u)-backprojection;first_inputs=[np.stack(u).copy()];first_gradients=[g.copy()];history=[];energies=[]
 for iteration in range(1,nitermax+1):
  delta_square=0.;u_square=0.
  for j in range(len(u)):
   p[j]+=(sigma*weight)*grad2d(ubar[j]);p[j]/=np.maximum(np.sqrt((p[j]**2).sum(axis=0)),1)
   ubar[j]=-u[j];u[j]+=tau*(weight*div2d(p[j])-g[j])
   if positivity:u[j]=np.maximum(u[j],0)
   ubar[j]+=2*u[j]
   delta_square+=float(np.sum((ubar[j]-u[j])**2));u_square+=float(np.sum(u[j]**2))
  relative=math.sqrt(delta_square/u_square) if u_square else 0.
  if iteration==1 or iteration%20==0:
   history.append((iteration,relative))
   if energy is not None:energies.append((iteration,float(energy(u))))
  if iteration>=nitermax or delta_square<tol**2*u_square:break
  g=normal(u)-backprojection
  if len(first_inputs)<4:first_inputs.append(np.stack(u).copy());first_gradients.append(g.copy())
 return np.stack(u),dict(iterations=iteration,relative_update=relative,first_iterate_inputs=np.stack(first_inputs),first_iterate_gradients=np.stack(first_gradients),history=np.array(history),energy_history=np.array(energies).reshape(-1,2),tau=tau,sigma=sigma)

def reconstruct(observations,model,lbda=10.,tol=1e-5,nitermax=5000,positivity=False,track_energy=False):
 kernels=model.kernels();normal=NormalOperator(kernels,model.shape);v=model.adjoint(observations)
 init=np.array([x/(model.delta**2*x.sum()) for x in v])
 if positivity:init=np.maximum(init,0)
 Lf=lipschitz_power_iteration(init,normal,tol=tol)
 weight=max(float(x.max()-x.min()) for x in model.spectra)*model.nproj/(2*math.pi)*model.nb/(model.field[-1]-model.field[0])*model.delta*lbda
 def objective(x):
  x=np.asarray(x);quadratic=.5*np.sum(x*normal(x))-np.sum(x*v)+.5*np.sum(observations**2)
  return quadratic+weight*sum(np.sqrt((grad2d(image)**2).sum(axis=0)).sum() for image in x)
 u,diagnostics=primal_dual(init,normal,v,Lf,weight,tol=tol,nitermax=nitermax,positivity=positivity,energy=objective if track_energy else None)
 diagnostics.update(kernels=kernels,backprojected_observations=v,initialization=init,lipschitz=Lf,unnormalized_tv_weight=weight)
 return u,diagnostics

def field_metrics(images,truth):
 """Per-species cosine NCC, dynamic-range NRMSE and integrated-amount error."""
 result=[]
 for a,t in zip(images,truth):
  result.append([float(np.vdot(a,t)/(np.linalg.norm(a)*np.linalg.norm(t))),float(np.sqrt(np.mean((a-t)**2))/np.ptp(t)),float(abs(a.sum()-t.sum())/abs(t.sum()))])
 return np.array(result)

def review_comparisons(data,primary,diagnostics,truth,clean_observations):
 """Fresh maintenance ablations; primary data/model/thresholds remain fixed.

The uncoupled comparison minimizes sum_j .5||A_j u_j-y||²+W TV(u_j),
which fits the mixture independently using each species and exposes cross-talk.
Noise cases share one realization; they are not a bootstrap coverage study.
 """
 from .preprocessing import make_model
 from .physics_model import EPRModel
 from .generate_data import noise_variants
 normal=NormalOperator(diagnostics['kernels'],primary.shape[-2:]);v=diagnostics['backprojected_observations'];init=diagnostics['initialization'];Lf=diagnostics['lipschitz'];weight=diagnostics['unnormalized_tv_weight']
 methods=[primary];method_names=['Signed TV (primary)']
 for name,factor,positive in [('Nonnegative TV',1.,True),('Weak TV',.25,False),('Strong TV',4.,False)]:
  u,_=primal_dual(init,normal,v,Lf,weight*factor,positivity=positive);methods.append(u);method_names.append(name)
 independent_kernels=diagnostics['kernels'].copy();independent_kernels[0,1]=0;independent_kernels[1,0]=0
 independent_normal=NormalOperator(independent_kernels,primary.shape[-2:]);independent_L=lipschitz_power_iteration(init,independent_normal)
 separate,_=primal_dual(init,independent_normal,v,independent_L,weight);methods.extend([separate,init]);method_names.extend(['Ignore source coupling','Normalized backprojection'])
 primary_model=make_model(data)
 method_residuals=np.array([np.sqrt(np.mean((primary_model.project(u)-data['projections'])**2))/5000. for u in methods])
 noise_scales=np.array([.5,1.,2.]);fixed_noise=data['projections']-clean_observations
 observations=noise_variants(clean_observations,fixed_noise,noise_scales);noise_fields=[];noise_metrics=[]
 for signal in observations:
  rhs=primary_model.adjoint(signal);start=np.array([x/(primary_model.delta**2*x.sum()) for x in rhs]);lip=lipschitz_power_iteration(start,normal)
  u,_=primal_dual(start,normal,rhs,lip,weight);noise_fields.append(u);noise_metrics.append(field_metrics(u,truth))
 angular_counts=np.array([10,20,40]);angular_fields=[];angular_metrics=[]
 for count in angular_counts:
  if count==40:u=primary.copy()
  else:
   step=40//int(count);model=EPRModel(data['magnetic_field_G'],data['gradients_G_cm'][:,::step],data['reference_spectra']);u,_=reconstruct(data['projections'][::step],model)
  angular_fields.append(u);angular_metrics.append(field_metrics(u,truth))
 result=dict(method_reconstructions=np.stack(methods),method_metrics=np.array([field_metrics(u,truth) for u in methods]),method_residual_sigma=method_residuals,noise_scales=noise_scales,noise_reconstructions=np.stack(noise_fields),noise_metrics=np.stack(noise_metrics),angular_counts=angular_counts,angular_reconstructions=np.stack(angular_fields),angular_metrics=np.stack(angular_metrics))
 return result,method_names
