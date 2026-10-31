from pathlib import Path
import numpy as np
from src.preprocessing import load_data,analysis_mask
from src.solvers import reconstruct
from src.physics_model import predict_from_parameters
ROOT=Path(__file__).resolve().parents[2]
def test_original_pipeline():
    d=load_data(ROOT/'data/raw_data.npz');r=reconstruct(d)
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(r['reconstruction'],f['output_mtsat'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(r['r1'],f['output_r1'],atol=1e-10,rtol=1e-10)
    mask=analysis_mask(d);assert mask.sum()==219351
    _,pred=predict_from_parameters(d,r['reconstruction'][0],r['r1'][0],mask)
    obs=np.stack([d[k][mask] for k in ('PDw','T1w','MTw')])
    np.testing.assert_allclose(pred,obs,rtol=1e-12,atol=1e-10)
