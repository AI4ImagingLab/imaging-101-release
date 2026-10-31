from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_observations,preprocess,COORDS
from src.physics_model import fit_covariance,normalization,grid_points,projected_coordinates,probability
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_profile_processing_covariance_and_corrected_modes():
    raw=load_observations(ROOT/'data/raw_data.npz')
    measured,moments=preprocess(raw)
    f=np.load(ROOT/'test/fixtures/upstream_preprocessing.npz',allow_pickle=False)
    for j,d in enumerate(['x','y','u']):
        for k in range(8):assert_allclose(measured[k,j],f[f'profile_{d}_{k}'][1],rtol=1e-9,atol=1e-10)
    S=fit_covariance(moments,raw['transfer_matrices'])
    N=normalization(S)
    assert_allclose(S,f['covariance'],rtol=1e-9,atol=1e-10)
    assert_allclose(N,f['normalization'],rtol=1e-9,atol=1e-10)
    assert_allclose(N@S@N.T,np.eye(4),rtol=1e-12,atol=1e-12)

def test_distinct_emittances_detect_original_scale_order_error():
    # Fixed independent SPD cases, not simulated benchmark observations.
    base=np.diag([4.,9.,25.,36.])
    T=np.array([[1,0,.3,0],[0,1,0,0],[0,0,1,0],[0,-.3,0,1.]])
    for S in [base,T@base@T.T]:
        N=normalization(S)
        assert_allclose(N@S@N.T,np.eye(4),rtol=1e-12,atol=1e-12)

def test_independent_covariance_inverse_known_spd():
    raw=load_observations(ROOT/'data/raw_data.npz')
    A=np.array([[2,0,.2,0],[.4,1,0,.1],[0,0,3,0],[.2,0,.5,1.2]])
    S=A@A.T
    directions=np.array([[1,0,0,0],[0,0,1,0],[1,0,-1,0]])/np.array([[1],[1],[np.sqrt(2)]])
    moments=[]
    for M in raw['transfer_matrices']:
        moments.append([d@M@S@M.T@d for d in directions])
    assert_allclose(fit_covariance(np.asarray(moments),raw['transfer_matrices']),S,rtol=1e-10,atol=1e-10)

def test_deterministic_first_projection_and_update_with_frozen_particles(monkeypatch):
    import src.solvers as solver
    raw=load_observations(ROOT/'data/raw_data.npz')
    fixed=np.load(ROOT/'test/fixtures/upstream_fixed_particles.npz',allow_pickle=False)
    # Isolate deterministic operators by supplying actual frozen upstream
    # particles. This is not a reduced benchmark or a stochastic trajectory test.
    monkeypatch.setattr(solver,'sample_grid',lambda probability,size,rng:fixed['particles'])
    result=solver.reconstruct(raw,epochs=1,seed=None)
    fixture=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    assert_allclose(result['first']['probability'],fixture['initial_grid_probability'],rtol=1e-9,atol=1e-10)
    assert_allclose(result['first']['simulated'],fixed['histogram'],rtol=1e-9,atol=1e-10)
    assert_allclose(result['first']['updated_function'],fixed['updated_function'],rtol=1e-9,atol=1e-10)
