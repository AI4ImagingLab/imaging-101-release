from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_fresh_all_band_fit_and_original_parity():
    result,diag=reconstruct(load_data(ROOT/'data/raw_data.npz'),return_diagnostics=True)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:
        for i in range(3):
            p=result['parameters'][0,i];q=f['parameters'][0,i]
            assert np.max(abs(p[:2]-q[:2]))<.02
            assert abs(p[2]-q[2])/q[2]<.005
            assert np.max(abs(p[3:7]-q[3:7]))<.005
            assert abs(p[7]-q[7])<2e-6
            assert np.sqrt(np.mean((result['reconstruction'][0,i]-f['reconstruction'][0,i])**2))/np.ptp(f['reconstruction'][0,i])<.001
    assert all(np.isfinite(h).all()and h[-1]<h[0]for h in diag['history'])
