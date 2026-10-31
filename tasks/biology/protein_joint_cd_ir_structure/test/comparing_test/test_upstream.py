from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_all_stages_exact_bound():
    result,details=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        for i,d in enumerate(details):
            for key in ['hj5','ix','sel1','sel2','sel3','Mean_refit_prot','refit_np_param','attempts','refit_rule']:
                np.testing.assert_allclose(d[key],f[f'output_scale15_target{i}_{key}'],atol=1e-9,rtol=1e-9)
def test_all_scientific_comparisons():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as f:
        for scale in [0,1,5,10,15,20,-1]:
            r,d=reconstruct(data,ir_scale=scale if scale>=0 else 15,mode='ir' if scale==-1 else 'joint')
            for i,row in enumerate(d):
                np.testing.assert_allclose(row['fractions'],f[f'scale{scale}_target{i}_sel3'],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(row['spectrum'],f[f'scale{scale}_target{i}_Mean_refit_prot'],atol=1e-9,rtol=1e-9)
