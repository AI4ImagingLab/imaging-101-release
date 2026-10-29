from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct
from src.physics_model import phase_consistency

ROOT=Path(__file__).resolve().parents[2]
def test_complete_exact_bound_rf_pipeline():
    result=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        roi=f['roi'];reference=f['smoothed_vector']
        np.testing.assert_allclose(result['autocorrelation'],f['autocorrelation'],rtol=2e-9,atol=2e-4)
        np.testing.assert_allclose(result['raw_velocity'],f['raw_vector'],rtol=2e-9,atol=2e-9)
        np.testing.assert_allclose(result['velocity'][:,roi],reference[:,roi],rtol=2e-10,atol=2e-9)
    assert np.all(result['velocity'][:,~roi]==0)
    assert abs(phase_consistency(result['velocity'],result['autocorrelation'],roi)-.05201278063077122)<2e-12
    assert len(result['smoothing']['history'])==3
    # The original exitflag describes its final robust pass, not every inner
    # pass. Its first pass reaches100 updates before subsequent reweighting.
    assert result['smoothing']['history'][-1]['converged']
