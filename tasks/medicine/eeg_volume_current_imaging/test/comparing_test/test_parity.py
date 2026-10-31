from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
from src.physics_model import forward
ROOT=Path(__file__).resolve().parents[2]
def test_original_mne_current_kernel_and_projection():
    data=load_data(ROOT/'data/raw_data.npz');out,stage=reconstruct(data)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:np.testing.assert_allclose(out['reconstruction'],f['reconstruction'],rtol=2e-10,atol=1e-20)
    with np.load(ROOT/'test/fixtures/oracle_stages.npz') as f:
        # SVD bases differ at cancelling tiny kernel entries; compare the
        # independently assembled operator in its scale-aware matrix norm.
        assert np.linalg.norm(stage['kernel']-f['output_kernel'])/np.linalg.norm(f['output_kernel'])<1e-10
        np.testing.assert_allclose(stage['projection'],f['output_projection'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(stage['singular_values'],f['output_singular_values'],rtol=1e-7,atol=1e-13)
        np.testing.assert_allclose(f['output_source_covariance'],1/stage['trace_scale'],rtol=1e-13)
        # Original MNE whitens into a covariance-eigenvector basis; our
        # symmetric whitening is equivalent under an orthogonal basis change.
        w=f['output_whitener']
        np.testing.assert_allclose(w.T@w,stage['projection']/1e-10,rtol=1e-12,atol=1e-4)
def test_bound_measurements_reconstruction_improves_forward_consistency():
    d=load_data(ROOT/'data/raw_data.npz');out,_=reconstruct(d);x=out['reconstruction'][0]
    target=d['measurements']-d['measurements'].mean(axis=0)
    error=np.linalg.norm(forward(x,d['lead_field'])-target)/np.linalg.norm(target)
    assert .30<error<.55
    assert np.linalg.norm(x)>0
