from pathlib import Path
import numpy as np
from src.preprocessing import load_data,droplets
from src.physics_model import *
from src.solvers import reconstruct,correlation_signs,objective,optimization_inputs
ROOT=Path(__file__).resolve().parents[2]

def test_basis_orthogonality_and_geometry():
    d=load_data(ROOT/'data/raw_data.npz');b=basis(d['polar_angle'],d['azimuth']);w=d['quadrature_weight']
    np.testing.assert_allclose((b*w)@b.T,np.eye(4)/(32*np.pi),atol=1e-12,rtol=0)
    assert abs(w.sum()-1)<1e-12

def test_ideal_acquisition_recovers_unknown_process():
    d=load_data(ROOT/'data/raw_data.npz')
    for q in [np.array([.2,.3,.4,.5]),np.array([-.5,.2,-.4,.3]),np.array([.8,-.4,.3,.1])]:
        q=q/np.linalg.norm(q);u=quaternion_matrix(q)
        e=ideal_expectation(q,d['polar_angle'],d['azimuth'])
        c=overlap_coefficients(droplets(e),basis(d['polar_angle'],d['azimuth']),d['quadrature_weight'])
        for j in range(4):np.testing.assert_allclose(coefficients_matrix(c[j]),q[j]*u,atol=1e-12)
        fresh=reconstruct(dict(d,expectation=e))['reconstruction'][0]
        np.testing.assert_allclose(fresh,u,atol=1e-12)
        np.testing.assert_allclose(u.conj().T@u,np.eye(2),atol=1e-12)

def test_objective_directional_derivative():
    d=load_data(ROOT/'data/raw_data.npz');channels,x=optimization_inputs(d)
    # Independent derivative via complex matrix/quaternion Jacobian formed from centered differences,
    # checked against scalar-objective finite differences at a different step.
    direction=np.array([.3,-.2,.4]);h=1e-5;k=1e-6
    def model(z):
        q=euler_quaternion(z);return q[:,None,None]*quaternion_matrix(q)
    residual=model(x)-channels
    jac=(model(x+k*direction)-model(x-k*direction))/(2*k)
    derivative=np.real(np.vdot(residual,jac))
    difference=(objective(x+h*direction,channels)-objective(x-h*direction,channels))/(2*h)
    assert abs(derivative-difference)<1e-6

def test_signs_and_global_scale():
    d=load_data(ROOT/'data/raw_data.npz');a=reconstruct(d)['reconstruction']
    np.testing.assert_allclose(reconstruct(dict(d,expectation=d['expectation']*.4))['reconstruction'],a,atol=1e-12)
    for k in range(4):
        q=np.array([.2,-.3,.4,1.]);q[k]*=20
        np.testing.assert_array_equal(correlation_signs(np.outer(q,q)),np.sign(q))

def test_public_contract():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert 'github' not in text.lower()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        for k in z.files:assert ('| '+k+' |') in text and z[k].shape[0]==1
    assert '50' in text and 'dimensionless' in text and '(1, 2, 2)' in text
