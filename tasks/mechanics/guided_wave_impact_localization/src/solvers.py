import time
import numpy as np
import torch
from .preprocessing import prepare
from .physics_model import GP

def fit_model(x,y,iterations=5000,seed=0):
 torch.manual_seed(seed);model=GP();optimizer=torch.optim.Adam(model.parameters(),lr=.1);trace=[]
 for i in range(iterations):
  optimizer.zero_grad();loss=model.nll(x,y);loss.backward();optimizer.step();trace.append(float(loss.detach()))
 return model,np.asarray(trace,dtype=np.float64)

def reconstruct(data,normalization='sample',sensors=None,iterations=5000,seed=0):
 x,y,z=prepare(data,normalization,sensors);model,trace=fit_model(x,y,iterations,seed)
 with torch.no_grad():mean,covariance=model.predict(x,y,z)
 result={'reconstruction':mean.numpy().reshape(1,3,35,2).astype(np.float64)}
 return result,dict(model=model,train_x=x,train_y=y,test_x=z,covariance=covariance.numpy(),loss_history=trace)
