import numpy as np
from src.physics_model import objective_gradient
from src.solvers import solve_sample
def test_gradient():
 a=np.array([[1.,2.],[3.,4.]]);r=np.array([[0.,-1.]]);q=np.array([.3,.7,.2]);y=np.array([2.,3.]);w=np.array([2.,.5]);p=np.array([.1,.2])
 v,g=objective_gradient(q,y,a,r,w,p)
 for j in range(3):
  e=np.eye(3)[j]*1e-6;fd=(objective_gradient(q+e,y,a,r,w,p)[0]-objective_gradient(q-e,y,a,r,w,p)[0])/2e-6
  assert np.isclose(fd,g[j],rtol=2e-8,atol=2e-9)
def test_analytic_conserved_mixture():
 q,it=solve_sample(np.array([.25,.75]),np.eye(2),np.zeros((0,2)),np.ones(2),np.zeros(2));assert np.allclose(q,[.25,.75],atol=2e-9)
