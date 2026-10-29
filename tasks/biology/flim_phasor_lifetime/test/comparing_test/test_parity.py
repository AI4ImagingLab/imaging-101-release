from pathlib import Path
import numpy as np
from src.physics_model import normalized_phasor,calibrate,lifetimes
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_stages():
    f=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    mean,g,s=normalized_phasor(f['input_signal'])
    for actual,name in [(mean,'mean'),(g,'real'),(s,'imag')]:
        np.testing.assert_allclose(actual,f['output_'+name],rtol=1e-10,atol=1e-12)
    G,S=calibrate(g,s,f['input_calibration'])
    np.testing.assert_allclose(G,f['output_calibrated_real'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(S,f['output_calibrated_imag'],rtol=1e-10,atol=1e-12)
    a,b=lifetimes(G,S)
    for actual,name in [(a,'phase'),(b,'modulation')]:
        np.testing.assert_allclose(actual,f['output_'+name],rtol=1e-10,atol=1e-12,equal_nan=True)
def test_pipeline():
    result=reconstruct(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
        for k in ref.files: np.testing.assert_allclose(result[k],ref[k],rtol=1e-10,atol=1e-12)
