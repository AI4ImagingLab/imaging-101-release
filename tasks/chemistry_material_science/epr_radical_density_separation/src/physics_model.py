"""Explicit local EPR mathematics; only NumPy/SciPy numerical primitives.

Adapted from MIT PyEPRI, Copyright(c)2024 Remy Abergel. The source oracle
and FINUFFT are NOT imported here. Nonuniform transforms are direct separable
exponential sums; primal-dual iterations preserve the author CP2016 updates.
"""
import math
import numpy as np

def grad2d(u):
 g=np.zeros((2,)+u.shape,dtype=u.dtype);g[0,:-1]=u[1:]-u[:-1];g[1,:,:-1]=u[:,1:]-u[:,:-1];return g

def div2d(p):
 out=np.zeros(p.shape[1:],dtype=p.dtype)
 out[1:-1]=p[0,1:-1]-p[0,:-2];out[0]=p[0,0];out[-1]=-p[0,-2]
 out[:,1:-1]+=p[1,:,1:-1]-p[1,:,:-2];out[:,0]+=p[1,:,0];out[:,-1]-=p[1,:,-2]
 return out

class EPRModel:
 def __init__(self,field,gradients,spectra,delta=.03,shape=(100,100)):
  self.field=np.asarray(field,dtype=np.float64);self.gradients=np.asarray(gradients,dtype=np.float64);self.spectra=np.asarray(spectra,dtype=np.float64);self.delta=float(delta);self.shape=tuple(shape)
  self.nb=len(field);self.nproj=gradients.shape[1];self.half=1+self.nb//2
  alpha=np.arange(self.half,dtype=np.float64);db=self.field[1]-self.field[0]
  mu=np.sqrt((self.gradients**2).sum(axis=0))[:,None]
  self.valid=(mu*alpha[None,:]<.5*self.nb*db/self.delta)&(alpha[None,:]<.5*self.nb)
  self.indexes=np.flatnonzero(self.valid.ravel())
  xi=2*math.pi*alpha/(self.nb*db)
  self.xnodes=(-(self.delta*self.gradients[0])[:,None]*xi[None,:]).ravel()[self.indexes]
  self.ynodes=(-(self.delta*self.gradients[1])[:,None]*xi[None,:]).ravel()[self.indexes]
  self.hfft=np.fft.rfft(self.spectra,axis=-1)
  self.node_projection,self.node_frequency=np.unravel_index(self.indexes,(self.nproj,self.half))
  self._phases={}
 def phases(self,shape):
  shape=tuple(shape)
  if shape not in self._phases:
   # Centered physical lattice. Arrays are mode x irregular-node for efficient
   # separable products, with no opaque NUFFT/interpolation approximation.
   ky=np.arange(shape[0])-shape[0]//2;kx=np.arange(shape[1])-shape[1]//2
   self._phases[shape]=(np.exp(-1j*self.ynodes[:,None]*ky[None,:]),np.exp(-1j*self.xnodes[:,None]*kx[None,:]))
  return self._phases[shape]
 def type2(self,u):
  ey,ex=self.phases(u.shape)
  return np.sum(ey*(u@ex.T).T,axis=1)
 def type1(self,c,shape):
  ey,ex=self.phases(shape)
  return ey.conj().T@(c[:,None]*ex.conj())
 def project(self,u):
  out=np.zeros((self.nproj,self.half),dtype=np.complex128)
  for j,image in enumerate(u):
   coefficients=np.zeros_like(out);coefficients.ravel()[self.indexes]=self.type2(image)
   coefficients*=self.delta**2*self.hfft[j][None,:]
   out+=coefficients
  return np.fft.irfft(out,n=self.nb,axis=-1)
 def adjoint(self,observations):
  v=np.fft.rfft(observations,axis=-1);result=[]
  for spectrum in self.hfft:
   c=v*spectrum.conj()[None,:];c[:,0]*=.5
   z=self.type1(c.ravel()[self.indexes],self.shape)
   result.append((z+z.conj()).real*(self.delta**2/self.nb))
  return np.stack(result)
 def kernels(self):
  shape=tuple(2*s for s in self.shape);result=np.empty((2,2)+shape)
  for k in range(2):
   for j in range(2):
    c=np.tile((self.hfft[j]*self.hfft[k].conj())[None,:],(self.nproj,1));c[:,0]*=.5
    z=self.type1(c.ravel()[self.indexes],shape)
    result[k,j]=(z+z.conj()).real*(self.delta**4/self.nb)
  return result

class NormalOperator:
 def __init__(self,kernels,shape):
  self.shape=tuple(shape);self.padded=tuple(2*s for s in shape);self.spectra=np.fft.rfft2(kernels,axes=(-2,-1))
 def __call__(self,u):
  transforms=[np.fft.rfft2(image,s=self.padded) for image in u]
  output=[]
  for k in range(len(u)):
   image=np.zeros(self.shape)
   for j in range(len(u)):
    image+=np.fft.irfft2(transforms[j]*self.spectra[k,j],s=self.padded)[self.shape[0]:,self.shape[1]:]
   output.append(image)
  return np.stack(output)

def review_physics(model,normal,observations):
 """Actively execute analytical point, Euclidean adjoint, TV and FD checks."""
 rng=np.random.default_rng(864021)
 u=rng.normal(size=(2,)+model.shape);v=rng.normal(size=observations.shape)
 Au=model.project(u);ATv=model.adjoint(v);lhs=float(np.vdot(Au,v));rhs=float(np.vdot(u,ATv))
 direct=model.adjoint(Au);fast=normal(u)
 point=np.zeros((2,)+model.shape);point[0,43,61]=2.
 prediction=model.project(point);alpha=np.arange(model.half);omega=2*np.pi*alpha/(model.nb*(model.field[1]-model.field[0]))
 shift=model.gradients[0]*(11*model.delta)+model.gradients[1]*(-7*model.delta)
 expected=np.fft.irfft(2*model.delta**2*model.hfft[0][None,:]*np.exp(1j*shift[:,None]*omega[None,:])*model.valid,n=model.nb,axis=1)
 scalar=rng.normal(size=(17,23));vector=rng.normal(size=(2,17,23));tv_l=float(np.vdot(grad2d(scalar),vector));tv_r=float(-np.vdot(scalar,div2d(vector)))
 direction=rng.normal(size=u.shape);eps=1e-5;objective=lambda x:.5*np.sum((model.project(x)-observations)**2)
 fd=(objective(u+eps*direction)-objective(u-eps*direction))/(2*eps);analytic=float(np.vdot(model.adjoint(model.project(u)-observations),direction))
 names=['Point shift','Adjoint dot','Toeplitz normal','TV adjoint','Objective derivative']
 errors=np.array([np.linalg.norm(prediction-expected)/np.linalg.norm(expected),abs(lhs-rhs)/max(abs(lhs),abs(rhs)),np.linalg.norm(fast-direct)/np.linalg.norm(direct),abs(tv_l-tv_r)/max(abs(tv_l),abs(tv_r)),abs(fd-analytic)/max(abs(fd),abs(analytic))])
 return dict(point_predicted=prediction,point_expected=expected,normal_direct=direct,normal_fft=fast,adjoint_dot_values=np.array([lhs,rhs]),tv_dot_values=np.array([tv_l,tv_r]),objective_derivative_values=np.array([fd,analytic]),errors=errors),names

