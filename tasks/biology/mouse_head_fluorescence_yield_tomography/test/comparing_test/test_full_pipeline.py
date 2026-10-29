from pathlib import Path
import numpy as np
from src.preprocessing import load_raw
from src.solvers import reconstruct
T=Path(__file__).resolve().parents[2]
def test_complete_frozen_reference_and_original_stop_rule():
    result=reconstruct(load_raw(T/'data/raw_data.npz'))
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as f:reference=f['reconstruction'][0]
    x=result['reconstruction'];assert x.shape==(5707,) and np.isfinite(x).all() and np.all(x>=0)
    assert x@reference/np.linalg.norm(x)/np.linalg.norm(reference)>=.999999
    assert np.sqrt(np.mean((x-reference)**2))/np.ptp(reference)<=1e-6
    assert np.max(abs(x-reference))<=1e-10
    assert result['forward_relative_L2']<=.191
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        assert np.allclose(result['objective_history'],f['projection_error'],rtol=1e-9,atol=1e-9)
        assert result['initial_muaf']==float(f['uniform_muaf'])
    h=result['objective_history'];assert np.all(np.diff(h)<0)
    assert 100*(h[-2]-h[-1])/h[-2]<=2
    assert np.all(100*(h[:-2]-h[1:-1])/h[:-2]>2)
