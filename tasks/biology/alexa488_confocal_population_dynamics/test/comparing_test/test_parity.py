from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.physics_model import correlation_model,objective
from src.preprocessing import load_observations
from src.solvers import reconstruct

def test_upstream_forward_and_data():
    a=np.load(ROOT/'test/fixtures/upstream.npz');t,g=load_observations(ROOT/'data/raw_data.npz');np.testing.assert_array_equal(np.column_stack([t,g]),a['input_observations'])
    p=a['output_parameters'][[0,1,3]]
    np.testing.assert_allclose(correlation_model(p,t),a['output_curve'],rtol=2e-15,atol=1e-17)
    np.testing.assert_allclose(correlation_model(p,np.array([0.,.001,.01,.1,1.])),a['output_probe'],rtol=2e-15)

def test_pipeline_parity():
    a=np.load(ROOT/'test/fixtures/upstream.npz');t,g=load_observations(ROOT/'data/raw_data.npz');p,m,sse=reconstruct(t,g)
    np.testing.assert_allclose(p,a['output_parameters'][[0,1,3]],rtol=1e-4)
    assert np.linalg.norm(m-a['output_curve'])/np.linalg.norm(m)<1e-6
    np.testing.assert_allclose(sse,objective(p,t,g),rtol=1e-15)

def test_analytic_physics():
    t=np.geomspace(1e-8,1e8,1000);p=[7.,.2,.03];m=correlation_model(p,t)
    assert np.all(m>0) and np.all(np.diff(m)<0)
    assert np.isclose(correlation_model(p,0),1/(7*.8))
    assert np.isclose(correlation_model([7.,0.,.03],0),1/7)
    assert np.isclose(m[-1]*t[-1]**1.5,5*.03**1.5/7,rtol=1e-7)

def test_readme_constants_and_contract():
    s=(ROOT/'README.md').read_text(encoding='utf-8');assert 'tauT=0.002 ms' in s and 'S=5' in s and '(1, 200)' in s
