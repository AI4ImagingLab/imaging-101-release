import numpy as np
import pytest
from src.physics_model import gradients,normal_equations,compose
from src.solvers import integrate_gradients
def test_masked_graph_adjoint_and_energy():
 rng=np.random.default_rng(144);m=np.ones((5,6),bool);m[1,2]=False;u=rng.normal(size=m.shape);ty,tx=gradients(u)
 wy=rng.uniform(.1,3,m.shape);wx=rng.uniform(.1,3,m.shape);A,b,n=normal_equations(ty,tx,m,wy,wx)
 np.testing.assert_allclose(A@u[m],b,atol=1e-13);np.testing.assert_allclose(A@np.ones(m.sum()),0,atol=1e-14)
 assert (A-A.T).nnz==0
 energy=np.sum(wy[1:][m[1:]&m[:-1]]*ty[1:][m[1:]&m[:-1]]**2)+np.sum(wx[:,1:][m[:,1:]&m[:,:-1]]*tx[:,1:][m[:,1:]&m[:,:-1]]**2)
 np.testing.assert_allclose(u[m]@(A@u[m]),energy,rtol=1e-14)
def test_constant_and_ramp():
 y,x=np.indices((4,7));u=.13*x+.27*y;m=np.ones(u.shape,bool);ty,tx=gradients(u)
 result,record=integrate_gradients(ty,tx,m);np.testing.assert_allclose(result,u-u.mean(),atol=1e-9);assert record['converged']
 np.testing.assert_array_equal(compose(np.full((2,3),2.),np.full((2,3),4.)),np.full((2,3),8.))
def test_disconnected_mask_explicit_failure():
 m=np.eye(3,dtype=bool)
 with pytest.raises(ValueError,match='connected'):integrate_gradients(np.zeros((3,3)),np.zeros((3,3)),m)
