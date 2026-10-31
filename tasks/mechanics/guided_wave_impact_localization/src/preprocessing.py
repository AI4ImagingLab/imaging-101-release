import numpy as np
import torch

def load(path="data/raw_data.npz"):
 with np.load(path,allow_pickle=False) as f:return {k:f[k][0] for k in f.files}

def prepare(data,normalization='sample',sensors=None):
 x=torch.tensor(data['calibration_tdoa'],dtype=torch.float32,device='cpu');z=torch.tensor(data['observations'].reshape(-1,6),dtype=torch.float32,device='cpu');y=torch.tensor(data['calibration_positions'],dtype=torch.float32,device='cpu')
 if sensors is not None:
  x=x[:,sensors];z=z[:,sensors];x=x-x.min(1,keepdim=True).values;z=z-z.min(1,keepdim=True).values
 if normalization=='sample':
  if torch.any(x.norm(dim=1)==0) or torch.any(z.norm(dim=1)==0):raise ValueError('zero TDOA direction')
  x=x/x.norm(dim=1,keepdim=True);z=z/z.norm(dim=1,keepdim=True)
 elif normalization=='feature':
  mean=x.mean(0);std=x.std(0,correction=1)
  if torch.any(std==0):raise ValueError('constant calibration feature')
  x=(x-mean)/std;z=(z-mean)/std
 elif normalization!='none':raise ValueError('unknown normalization')
 return x,y,z
