from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import forward,adjoint
from src.solvers import tv_denoise,reconstruct_burst,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_operators_tv_and_gap_admm_updates():
    data=load_data(ROOT/'data/raw_data.npz');y=data['observations'][0];mask=data['mask']
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(adjoint(y,mask),f['adjoint_measurement'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(forward(f['tv_input'],mask),f['forward_initial'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(tv_denoise(f['tv_input']),f['tv_output'],rtol=1e-10,atol=1e-10)
        for count in [1,2,5]:
            for method in ['gap','admm']:
                value,_=reconstruct_burst(y,mask,method=method,iterations=count)
                np.testing.assert_allclose(value,f[f'{method}_{count}'],rtol=1e-10,atol=1e-10)

def test_original_complete15bursts210frames():
    result,diagnostics=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(result['reconstruction'],f['reconstruction'],rtol=0,atol=1e-7)
    assert len(diagnostics)==15 and all(d['iterations']==100 for d in diagnostics)
    # Independent original repeats retain internal mask order; odd chronological
    # burst index7 is reversed by the wrapper, rather than by the evaluator.
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for index,repeat in zip([0,7,14],f['independent_repeat']):
            original=repeat if index%2==0 else repeat[::-1]
            np.testing.assert_allclose(result['reconstruction'][0,index*14:(index+1)*14],original,rtol=0,atol=1e-7)
