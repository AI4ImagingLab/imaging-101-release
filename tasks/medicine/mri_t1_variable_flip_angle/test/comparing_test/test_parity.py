from pathlib import Path
import numpy as np
from src.physics_model import spgr_signal,linearized_fit
from src.solvers import reconstruct,reconstruct_uncorrected
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_stages():
    f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    t1,m0=linearized_fit(f['input_signal'],f['input_b1'])
    np.testing.assert_allclose(np.stack([t1,m0],axis=-1),f['output_parameters'],rtol=1e-10,atol=1e-8)
    signal=spgr_signal(f['input_parameters'][:,0],f['input_parameters'][:,1],f['input_b1'])
    np.testing.assert_allclose(signal,f['output_forward'],rtol=1e-10,atol=1e-9)
def test_pipeline():
    result=reconstruct(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
        for k in ref.files:np.testing.assert_allclose(result[k],ref[k],rtol=1e-10,atol=1e-8)
    ref=np.load(ROOT/'notebooks/reference_outputs/upstream_uncorrected.npz')['reconstruction']
    np.testing.assert_allclose(reconstruct_uncorrected(ROOT/'data/raw_data.npz'),ref,rtol=1e-10,atol=1e-8)
