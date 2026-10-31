"""Explicit dense/ELU forward response and input-gradient chain rule; GPL-3.0."""
import numpy as np

def forward(normalized, data, target=None):
 x=np.asarray(normalized,dtype=np.float32);states=[]
 for i in range(12):
  z=x@data[f'weight_{i}']+data[f'bias_{i}'];states.append(z)
  x=z if i==11 else np.where(z>0,z,np.expm1(np.minimum(z,0)))
 if target is None:return x
 residual=x[:,6:106]-target
 g=np.zeros_like(x);g[:,6:106]=residual
 for i in range(11,-1,-1):
  if i<11:g*=np.where(states[i]>0,1,np.exp(np.minimum(states[i],0)))
  g=g@data[f'weight_{i}'].T
 g=g.astype(np.float64);g[:,0]=0
 return residual,g

def normalized_parameters(parameters,data):
 physical=np.c_[data['solar_zenith'],parameters]
 return physical*data['scaler_scale']+data['scaler_offset']

def spectrum(parameters,data):return forward(normalized_parameters(parameters,data),data)

def step(q,gradient,accumulator):
 accumulator=accumulator+gradient*gradient
 return np.abs(q-gradient/np.sqrt(accumulator+1e-7)),accumulator
