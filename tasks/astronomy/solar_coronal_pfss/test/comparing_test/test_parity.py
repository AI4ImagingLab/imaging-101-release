from pathlib import Path
import numpy as np
from src.preprocessing import load_data,balance_flux
from src.solvers import solve
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_stages_and_all_components():
    raw=load_data(ROOT/'data/raw_data.npz');balanced=balance_flux(raw)
    fields,d=solve(balanced,return_diagnostics=True)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)as f:
        np.testing.assert_array_equal(balanced,f['output_balanced'])
        for k,v in d.items():np.testing.assert_allclose(v,f['output_'+k],rtol=1e-10,atol=1e-10)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)as f:
        for k,v in fields.items():np.testing.assert_allclose(v[None],f[k],rtol=1e-9,atol=1e-10)
