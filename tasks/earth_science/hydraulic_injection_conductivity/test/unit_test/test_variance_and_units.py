import numpy as np
from src.solvers import variance_diagnostics

def test_full_variance_known_saddle_inverse_and_source_omission():
    H=np.array([[1.],[2.]]);h=np.array([[.5],[-.5]]);Z=np.array([[.2],[.3],[.4]]);X=np.ones((3,1))/np.sqrt(3)
    source,complete=variance_diagnostics(H,h,Z,X,1.)
    A=np.block([[H@H.T+.25*np.eye(2),h],[h.T,np.zeros((1,1))]])
    expected=[];source_expected=[]
    inverse=np.linalg.inv(A);one_block=inverse.copy();one_block[:,-1]=0
    for z,x in zip(Z,X):
        b=np.r_[(H@z).ravel(),x];expected.append(1-b@inverse@b);source_expected.append(1-b@one_block@b)
    np.testing.assert_allclose(complete,expected,atol=1e-14,rtol=1e-14)
    np.testing.assert_allclose(source,np.minimum(1.,source_expected),atol=1e-14,rtol=1e-14)
    assert np.max(abs(source-complete))>.01

def test_constant_conductivity_analytic_recharge_profile():
    from scipy.sparse.linalg import splu
    from src.physics_model import HydraulicModel
    model=HydraulicModel(np.array([[5,9]]));A,rhs=model.assemble(np.full(7500,np.log(2.)),serialize=False)
    rhs[model.lookup[5*100+9],0]-=25.
    solution=splu(A).solve(rhs)[:,0]
    x=(model.active%100)*10.;L=990.
    expected=120-10*x/L+.001/(2*2.)*x*(L-x)
    np.testing.assert_allclose(solution,expected,atol=1e-8,rtol=0)
    assert np.max(abs(A@solution-rhs[:,0]))<1e-8
