from pathlib import Path
import numpy as np
from src.solvers import reconstruct
from src.preprocessing import load_data
ROOT=Path(__file__).resolve().parents[2]

def test_complete_qualified_source_reconstruction():
    result=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    fixture=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(result['corrected_observations'][::11,::9,::13],fixture['corrected_observation_probes'],rtol=3e-7,atol=1e-10)
    f=result['reconstruction'];reference=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'][0]
    assert f.shape==(77,55,80,15) and f.dtype==np.float64
    for j in range(15):
        x=f[...,j];r=reference[...,j];error=x-r
        assert np.max(abs(error))<=8e-9,(j,np.max(abs(error)))
        assert np.sqrt(np.mean(error**2))/np.ptp(r)<=1e-6
        assert np.vdot(x,r)/(np.linalg.norm(x)*np.linalg.norm(r))>=.999999
    assert result['relative_forward_residual']<=.2919
