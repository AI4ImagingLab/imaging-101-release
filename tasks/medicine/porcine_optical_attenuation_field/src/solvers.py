"""Local explicit depth-resolved confocal inverse; no OCT/domain library."""
import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import minimize
from .preprocessing import tissue_mask
from .physics_model import corrected_signal,DZ
def depth_inverse(term,dz=DZ):
 """Published Riemann inverse, excluding final depth from native denominator."""
 term=np.asarray(term,dtype=float);out=np.zeros_like(term)
 cumulative=np.cumsum(term[:-1][::-1],axis=0)[::-1]
 np.divide(term[:-1],2*dz*cumulative,out=out[:-1],where=cumulative!=0)
 return out
def reconstruct(signal,**kwargs):
 mask=tissue_mask(signal);term=corrected_signal(signal,mask,**kwargs)
 return depth_inverse(term),dict(mask=mask,term=term)
def native_vermeer(signal,mask):
 """Author comparison log1p convention (includes current sample), explicit."""
 a=np.asarray(signal)*mask;cum=np.cumsum(a[::-1],axis=0)[::-1];ratio=np.divide(a,cum,out=np.zeros_like(a),where=cum!=0)
 return np.log1p(ratio)/(2*DZ)
def gaussian_comparison(attenuation):
 """Native five-square Gaussian(sigma2), zero padding and first5rows restored."""
 x=np.arange(-2,3);g=np.exp(-(x[:,None]**2+x[None,:]**2)/8);g/=g.sum();out=convolve(attenuation,g,mode='constant',cval=0.)
 out[:5]=attenuation[:5];return out
def native_rof(attenuation,theta=1.):
 """Exact author legacy five-iteration comparison, including edge overwrites.

 Adapted Philippe Magiera/Carl Lundahl2008 routine distributed with author
 code. Five steps are a historical control, not a converged TV minimizer.
 """
 image=np.asarray(attenuation,dtype=float);p=np.zeros((*image.shape,2));div=np.zeros_like(image)
 for _ in range(5):
  div[1:-1]=p[1:-1,:,0]-p[:-2,:,0]
  div[:,1:-1]+=p[:,1:-1,1]-p[:,:-2,1]
  div[:,0]=p[:,0,1];div[:,-1]=-p[:,-2,1];div[0]=p[0,:,0];div[-1]=-p[-2,:,0]
  u=image-theta*div
  gy=np.vstack((np.diff(u,axis=0),np.zeros((1,image.shape[1]))));gx=np.column_stack((np.diff(u,axis=1),np.zeros(image.shape[0])))
  grad=np.stack((gy,gx),axis=-1);p=(p-(.25/theta)*grad)/(1+(.25/theta)*np.linalg.norm(grad,axis=-1,keepdims=True))
 return u
def horizontal_adjoint(dual):
 """Adjoint of forward column difference on one signal row, Neumann edge."""
 return np.r_[-dual[:1],dual[:-1]-dual[1:],dual[-1:]]
def intensity_weighted_tv(attenuation,masked_signal,mask,eta=6.,epsilon=.001):
 """Solve the distributed-code nonnegative horizontal weighted-TV variant.

 Every512-sample row is fit; CVX/domain functions are not imported. An explicit
 box-constrained dual uses generic L-BFGS-B and a measured primal-dual gap.
 The independent private convex-QP comparison is not used by this solver.
 """
 source=np.asarray(attenuation);out=source.copy();gaps=[];iterations=[]
 cleanup_sweeps=np.zeros(source.shape[0],dtype=np.int64)
 lbfgs_iterations=np.zeros(source.shape[0],dtype=np.int64)
 for j in range(source.shape[0]):
  count=int(np.count_nonzero(mask[j]))
  if not count:out[j]=0;continue
  b=source[j];gamma=(eta/count)*b/(np.asarray(masked_signal)[j]+epsilon);bounds=list(zip(-gamma[:-1],gamma[:-1]))
  if not np.any(gamma[:-1]>0):continue
  if np.ptp(b)<1e-12:out[j]=b.mean();continue
  def objective(p):
   value=np.maximum(b-horizontal_adjoint(p),0.)
   # Remove the data-only constant before convergence testing to avoid
   # cancellation/false relative convergence on bright constant signals.
   return .5*float(np.sum((value-b)*(value+b))),-np.diff(value)
  fit=minimize(objective,np.zeros(len(b)-1),jac=True,method='L-BFGS-B',bounds=bounds,
    options=dict(maxiter=20000,maxfun=40000,maxls=50,maxcor=20,ftol=0.,gtol=1e-10))
  p=fit.x.copy();raw=b-horizontal_adjoint(p)
  # Exact disjoint pair-coordinate minimizations finish dual convergence
  # when a generic line search stops before the independently checked gap.
  completed_sweeps=0
  for sweep in range(20001):
   if sweep%20==0:
    u=np.maximum(raw,0.);primal=.5*np.sum((u-b)**2)+np.sum(gamma[:-1]*np.abs(np.diff(u)))
    dual=.5*np.sum((b-u)*(b+u));gap=float(primal-dual)
    if abs(gap)<1e-9:break
   for parity in [0,1]:
    indices=np.arange(parity,len(p),2)
    proposed=np.clip(p[indices]+.5*(raw[indices+1]-raw[indices]),-gamma[indices],gamma[indices])
    change=proposed-p[indices];p[indices]=proposed;raw[indices]+=change;raw[indices+1]-=change
   completed_sweeps+=1
  out[j]=u
  relative=abs(gap)/max(1.,float(primal));gaps.append(relative);iterations.append(fit.nit)
  cleanup_sweeps[j]=completed_sweeps;lbfgs_iterations[j]=fit.nit
  if relative>1e-6:raise RuntimeError(f'iwhTV row{j}: primal-dual convergence gate failed, gap={relative:.6g}')
 return out,dict(max_relative_gap=max(gaps,default=0.),max_iterations=max(iterations,default=0),row_iterations=np.array(iterations),row_cleanup_sweeps=cleanup_sweeps,max_cleanup_sweeps=int(cleanup_sweeps.max()),row_lbfgs_iterations=lbfgs_iterations)
