from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import solve,coarse_to_fine,temporal_continuation
ROOT=Path(__file__).resolve().parents[2]

def test_complete_fresh_primary_against_frozen_original():
    result=solve(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_primary_stages.npz') as f:
        assert np.array_equal(result['index'],f['index'])
        assert np.array_equal(result['mask'],f['mask'])
        assert np.array_equal(result['normalized'],f['normalized'])
        mask=f['mask'];np.testing.assert_allclose(result['loss'][mask],f['loss'][mask],rtol=0,atol=2e-6*np.sqrt(3))
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:reference=f['reconstruction'][0]
    for i in range(3):
        d=result['reconstruction'][i];r=reference[i];m=mask[i]
        assert np.all(d[~m]==-1)
        assert np.max(np.abs(d[m]-r[m]))<=.001
        assert np.sqrt(np.mean((d[m].astype(float)-r[m])**2))/np.ptp(r[m])<=1e-6

def test_qualified_source_coarse_and_temporal_controls():
    raw=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_primary_stages.npz') as f:q=f['normalized'];mask=f['mask'];previous=f['index'][0]
    with np.load(ROOT/'notebooks/reference_outputs/upstream_controls.npz') as f:
        for i in range(3):
            c=coarse_to_fine(raw,q[i],mask[i]);np.testing.assert_allclose(c['reconstruction'],f[f'c2f_depth_{i}'],rtol=0,atol=2e-4)
            assert np.array_equal(c['relative_index'],f[f'c2f_relative_index_{i}'])
        for i in [1,2]:
            c=temporal_continuation(raw,q[i],mask[i],previous)
            np.testing.assert_allclose(c['reconstruction'],f[f'tc_depth_{i}'],rtol=0,atol=2e-4)
            assert np.array_equal(c['index'],f[f'tc_index_{i}']);previous=c['index']
