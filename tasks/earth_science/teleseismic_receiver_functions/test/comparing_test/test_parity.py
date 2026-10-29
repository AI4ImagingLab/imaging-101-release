from pathlib import Path
import numpy as np
from src.preprocessing import load_data,preprocess
from src.physics_model import rotate_zne
from src.solvers import waterlevel,time_domain,iterative
ROOT=Path(__file__).resolve().parents[2]
def test_original_stages_and_full_inverse():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        prepared=preprocess(load_data(ROOT/'data/raw_data.npz'));np.testing.assert_allclose(prepared,f['output_preprocessed'],rtol=1e-10,atol=1e-17)
        zrt=rotate_zne(prepared);np.testing.assert_allclose(zrt,f['output_rotated'],rtol=1e-10,atol=1e-17)
        rf,d=waterlevel(zrt);np.testing.assert_allclose(rf[None,1:],f['output_reconstruction'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(d['power'],f['output_water_power'],rtol=1e-10,atol=1e-25)
        np.testing.assert_allclose(d['source_conj'],f['output_source_spectrum_conj'],rtol=1e-10,atol=1e-16)
        np.testing.assert_allclose(d['filter'],f['output_filter'],rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(d['norm'],f['output_norm'],rtol=1e-10)
        np.testing.assert_allclose(time_domain(zrt)[1:],f['output_time'],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(iterative(zrt)[0][1:],f['output_iterative'],rtol=1e-9,atol=1e-10)
def test_full_pipeline():
    from src.main import reconstruct
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:np.testing.assert_allclose(reconstruct(ROOT/'data/raw_data.npz'),f['reconstruction'],rtol=1e-10,atol=1e-12)
