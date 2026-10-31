"""Explicit multitask RBF covariance, Gaussian likelihood and conditioning mathematics."""
import numpy as np
import torch
from torch.nn.functional import softplus
class GP(torch.nn.Module):
 def __init__(self,dtype=torch.float32):
  super().__init__();self.p=torch.nn.ParameterDict({})
  for k,v in [('task_noise',torch.zeros(2,dtype=dtype)),('global_noise',torch.zeros(1,dtype=dtype)),('factor',torch.randn(2,1,dtype=dtype)),('task_var',torch.randn(2,dtype=dtype)),('length',torch.zeros(1,1,dtype=dtype)),('mean0',torch.zeros((),dtype=dtype)),('mean1',torch.zeros((),dtype=dtype))]:self.p[k]=torch.nn.Parameter(v)
 def kernel(self,a,b):
  delta=(a[:,None,:]-b[None,:,:])/softplus(self.p['length']);return torch.exp(-.5*torch.sum(delta*delta,-1))
 def task(self):return self.p['factor']@self.p['factor'].T+torch.diag(softplus(self.p['task_var']))
 def noise(self):return softplus(self.p['task_noise'])+softplus(self.p['global_noise'])+2e-4
 def mean(self,n):return torch.stack([self.p['mean0'],self.p['mean1']]).repeat(n)
 def cov(self,x):return torch.kron(self.kernel(x,x).contiguous(),self.task().contiguous())+torch.diag(self.noise().repeat(len(x)))
 def nll(self,x,y):
  c=self.cov(x);L=torch.linalg.cholesky(c);r=y.flatten()-self.mean(len(x));a=torch.cholesky_solve(r[:,None],L)[:,0]
  return (.5*r@a+torch.log(torch.diag(L)).sum()+.5*r.numel()*np.log(2*np.pi))/r.numel()
 def predict(self,x,y,test):
  c=self.cov(x);L=torch.linalg.cholesky(c);r=y.flatten()-self.mean(len(x));cross=torch.kron(self.kernel(x,test).contiguous(),self.task().contiguous());a=torch.cholesky_solve(r[:,None],L)
  mean=self.mean(len(test))+cross.T@a[:,0];cov=torch.kron(self.kernel(test,test).contiguous(),self.task().contiguous())-cross.T@torch.cholesky_solve(cross,L)+torch.diag(self.noise().repeat(len(test)))
  return mean.reshape(-1,2),cov
