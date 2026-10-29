import numpy as np
from src.physics_model import mixing_matrix,rj_to_cmb,forward,adjoint
from src.solvers import separate,reconstruct
from src.preprocessing import load_data
def test_reference_frequency_laws_and_cmb_invariance():
    a=mixing_matrix([22.8,93.5]);assert a[0,1]==1 and a[1,2]==1
    np.testing.assert_array_equal(a[:,0],1.)
    assert np.all(rj_to_cmb(np.array([1.,10.,100.]))>1)
    assert abs(rj_to_cmb(.001)-1)<1e-9
def test_analytic_recovery_and_adjoint():
    a=mixing_matrix([22.8,33,40.7,60.8,93.5]);x=np.arange(18.).reshape(3,2,3)-7;y=np.arange(30.).reshape(5,2,3)
    d=forward(x,a);rec,cov,w=separate(d,a)
    np.testing.assert_allclose(rec,x,atol=2e-14)
    np.testing.assert_allclose(np.vdot(d,y),np.vdot(x,adjoint(y,a)),rtol=1e-13)
    np.testing.assert_allclose(w@a,np.eye(3),atol=2e-15)
    np.testing.assert_allclose(cov,np.linalg.inv(a.T@a),rtol=1e-13,atol=1e-13)
def test_normal_equation_stationarity_and_gain():
    d=load_data();a=mixing_matrix(d['frequencies']);m,p,c=reconstruct(d)
    scale=np.linalg.norm(a,2)*np.max(np.linalg.norm(d['measurements'],axis=0))
    assert np.max(np.abs(adjoint(p-d['measurements'],a)))/scale<5e-15
    np.testing.assert_allclose(separate(2*d['measurements'],a)[0],2*m,atol=2e-12)
def test_public_constants():
    from pathlib import Path
    text=Path('README.md').read_text(encoding='utf8');d=load_data()
    np.testing.assert_array_equal(d['frequencies'],[22.8,33,40.7,60.8,93.5])
    for value in ['2.7255','6.62607015e-34','1.380649e-23','1.54']:assert value in text
