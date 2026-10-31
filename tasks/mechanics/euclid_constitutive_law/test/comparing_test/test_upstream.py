from pathlib import Path
import numpy as np
from src.physics_model import features,weak_operator,frame_system,law
from src.preprocessing import load_data,element_deformation
from src.solvers import assemble,solve
ROOT=Path(__file__).resolve().parents[2]
def test_original_small_feature_and_weak_parity():
    z=np.load(ROOT/'test/fixtures/small_math_probe.npz');Q,G=features(z['F'])
    np.testing.assert_allclose(Q,z['features'],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(G,z['feature_gradient'],rtol=2e-12,atol=2e-12)
    tri=np.array([[0,1,2]]);grad=np.array([[[-1.,-1.],[1.,0.],[0.,1.]]]);A=weak_operator(tri,grad,np.array([.5]),G[:1],3)
    np.testing.assert_allclose(A.reshape(6,171),z['weak'],rtol=2e-12,atol=2e-12)
def test_actual_first_stage_features():
    z=np.load(ROOT/'test/fixtures/original_first_stage.npz');Q,_=features(z['output_F'])
    np.testing.assert_allclose(Q,z['output_features'],rtol=5e-9,atol=5e-13)
def test_full600_assembly_and_fresh_solution():
    data=load_data(ROOT/'data/raw_data.npz');H,b,C=assemble(data)
    z=np.load(ROOT/'test/fixtures/original_full_normal_system.npz')
    assert np.linalg.norm(H-z['output_LHS'])/np.linalg.norm(z['output_LHS'])<1e-12
    assert np.linalg.norm(b-z['output_RHS'])/np.linalg.norm(z['output_RHS'])<1e-12
    theta,info=solve(H,b,C);qualified=np.load(ROOT/'test/fixtures/corrected_optimizer.npz')['output_theta']
    W,P=law(data['query_gradients'],theta);Wr,Pr=law(data['query_gradients'],qualified)
    assert np.linalg.norm(W-Wr)/np.linalg.norm(Wr)<1e-5
    assert np.linalg.norm(P-Pr)/np.linalg.norm(Pr)<1e-5
    assert info['projected_kkt']<1e-8
