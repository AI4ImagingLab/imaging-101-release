from pathlib import Path
import sys
import numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load
from src.physics_model import forward,step

def test_chain_rule_finite_difference():
 d=load(T/'data/raw_data.npz')
 with np.load(T/'test/fixtures/upstream_stages.npz') as f:q=f['input_variables'][:1].copy()
 y=d['spectra'][:1,5::10][:,:100].astype(np.float32);r,g=forward(q,d,y)
 for k in range(1,6):
  h=1e-3;plus=q.copy();minus=q.copy();plus[0,k]+=h;minus[0,k]-=h
  rp=forward(plus,d)[:,6:106]-y;rm=forward(minus,d)[:,6:106]-y
  fd=(np.sum(rp.astype(float)**2)-np.sum(rm.astype(float)**2))/(4*h)
  np.testing.assert_allclose(g[0,k],fd,rtol=.015,atol=.003)

def test_adagrad_analytic():
 q=np.array([[1.,2.,3.]]);g=np.array([[0.,2.,-1.]]);acc=np.full_like(q,.1)
 actual,new=step(q,g,acc)
 np.testing.assert_allclose(new,[[.1,4.1,1.1]],rtol=0,atol=1e-15)
 np.testing.assert_allclose(actual,[[1.,abs(2-2/np.sqrt(4.1000001)),abs(3+1/np.sqrt(1.1000001))]],rtol=0,atol=1e-15)
