from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from src.physics_model import linear_polarizer,linear_retarder,acquisition,forward,adjoint,coherency_matrix
from src.solvers import reconstruct
from src.generate_data import author_example
ROOT=Path(__file__).resolve().parents[2]

def test_original_complete_native_cases():
    for n,rank in [(16,12),(24,16)]:
        with np.load(ROOT/f'test/fixtures/upstream_{n}.npz',allow_pickle=False) as z:
            out,d=reconstruct(z['angles'],z['power'])
            assert_allclose(d['generator'],z['generator'],rtol=0,atol=1e-14)
            assert_allclose(d['analyzer'],z['analyzer'],rtol=0,atol=1e-14)
            assert_allclose(d['design'],z['design'],rtol=0,atol=1e-14)
            assert_allclose(d['inverse'],z['inverse'],rtol=0,atol=1e-12)
            assert_allclose(d['singular_values'],z['singular_values'],rtol=0,atol=1e-14)
            assert_allclose(out['reconstruction'][0],z['reconstruction'],rtol=0,atol=1e-12)
            assert d['rank']==rank
            if n==24:assert_allclose(out['reconstruction'][0],z['truth'],rtol=0,atol=1e-12)
            else:assert np.linalg.norm(out['reconstruction'][0]-z['truth'])>.1

def test_local_author_generator_matches_independent_original():
    raw,truth=author_example()
    with np.load(ROOT/'test/fixtures/upstream_24.npz',allow_pickle=False) as z:
        assert_array_equal(raw['angles'][0],z['angles'])
        assert_allclose(raw['power'][0],z['power'],rtol=0,atol=1e-14)
        assert_allclose(truth,z['truth'],rtol=0,atol=1e-14)

def jones_mueller(jones):
    # Independent electromagnetic definition, rather than copied Mueller entries.
    p=np.array([[[1,0],[0,1]],[[1,0],[0,-1]],[[0,1],[1,0]],[[0,1j],[-1j,0]]],dtype=complex)
    return np.array([[.5*np.trace(p[a]@jones@p[b]@jones.conj().T).real for b in range(4)] for a in range(4)])

def test_independent_jones_optics_and_handedness():
    for angle in [0.,.2,.7,1.4]:
        v=np.array([np.cos(angle),np.sin(angle)])
        assert_allclose(linear_polarizer(angle),jones_mueller(np.outer(v,v)),rtol=1e-13,atol=1e-14)
        rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for delay in [0.,.3,np.pi/2,np.pi]:
            J=rot@np.diag(np.exp(np.array([-1j,1j])*delay/2))@rot.T
            assert_allclose(linear_retarder(angle,delay),jones_mueller(J),rtol=1e-13,atol=1e-14)
    assert_allclose(linear_retarder(0,np.pi/2)@np.array([1,0,1,0]),[1,0,0,-1],atol=1e-14)

def test_retarder_preserves_intensity_and_polarization():
    rng=np.random.default_rng(11);stokes=rng.normal(size=3);stokes/=np.linalg.norm(stokes);s=np.r_[1.,stokes]
    for angle,delay in rng.uniform(-3,3,size=(20,2)):
        M=linear_retarder(angle,delay);out=M@s
        assert_allclose(out[0],1,atol=1e-14)
        assert_allclose(np.linalg.norm(out[1:]),1,rtol=1e-14)
        assert_allclose(M.T@M,np.eye(4),atol=1e-14)

def test_stokes_cone_passivity_and_coherency():
    with np.load(ROOT/'data/raw_data.npz') as z:out,d=reconstruct(z['angles'][0],z['power'][0])
    M=out['reconstruction'][0];rng=np.random.default_rng(12);p=rng.normal(size=(1000,3));p/=np.linalg.norm(p,axis=1)[:,None]
    incident=np.column_stack((np.ones(1000),p));transmitted=incident@M.T
    assert transmitted[:,0].min()>=-1e-12 and transmitted[:,0].max()<=1+1e-12
    assert np.max(np.linalg.norm(transmitted[:,1:],axis=1)-transmitted[:,0])<1e-12
    h=coherency_matrix(M);assert_allclose(h,h.conj().T,atol=1e-14)
    assert np.linalg.eigvalsh(h).min()>-1e-12
    assert_allclose(np.linalg.eigvalsh(coherency_matrix(np.eye(4))),[0,0,0,1],atol=1e-14)
    assert_allclose(coherency_matrix(np.diag([1.,0,0,0])),np.eye(4)/4,atol=1e-14)

def test_forward_adjoint_and_objective_gradient():
    rng=np.random.default_rng(13);angles=np.linspace(0,np.pi,24);M=rng.normal(size=(4,4));y=rng.normal(size=24)
    assert_allclose(np.dot(forward(M,angles),y),np.sum(M*adjoint(y,angles)),rtol=1e-13)
    direction=rng.normal(size=(4,4));epsilon=1e-6
    loss=lambda X:.5*np.sum((forward(X,angles)-y)**2)
    numeric=(loss(M+epsilon*direction)-loss(M-epsilon*direction))/(2*epsilon)
    analytic=np.sum(direction*adjoint(forward(M,angles)-y,angles))
    assert_allclose(numeric,analytic,rtol=1e-7,atol=1e-8)

def test_full_rank_native_normal_equation_stationarity():
    with np.load(ROOT/'data/raw_data.npz') as z:out,d=reconstruct(z['angles'][0],z['power'][0])
    assert_allclose(d['design'].T@d['residual'],0,atol=1e-13)
    assert d['condition']<19 and d['rank']==16
