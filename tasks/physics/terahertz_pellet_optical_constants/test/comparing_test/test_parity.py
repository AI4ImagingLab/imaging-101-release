from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_stages_and_both_full_materials():
    t,e=load_data(ROOT/'data/raw_data.npz');r,d=reconstruct(t,e,return_diagnostics=True)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)as f:
        for key,ours in [('windowed',d['windowed']),('frequency',d['frequency_hz']*1e-12),('amplitude',d['amplitude']),('phase',d['phase'])]:np.testing.assert_allclose(ours,f['output_'+key],rtol=1e-12,atol=1e-12)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:
        for key in r:np.testing.assert_allclose(r[key],f[key],rtol=1e-11,atol=1e-11)
