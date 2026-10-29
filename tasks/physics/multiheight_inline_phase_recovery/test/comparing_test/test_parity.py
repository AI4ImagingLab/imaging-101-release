from pathlib import Path
import numpy as np
from src.preprocessing import amplitude_stack, load_data
from src.physics_model import wave_grid, transfer, propagate
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_stage_parity():
    f=np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)
    a=amplitude_stack(load_data(ROOT/'data/raw_data.npz'))
    np.testing.assert_array_equal(a,f['output_preprocessing'])
    np.testing.assert_array_equal(wave_grid((512,512)),f['output_kt2'])
    np.testing.assert_allclose(transfer((512,512),50),f['output_transfer50'],rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(propagate(a[0],50),f['output_step'],rtol=3e-6,atol=3e-6)
def test_full_upstream_parity():
    a=amplitude_stack(load_data(ROOT/'data/raw_data.npz'))
    out,_=reconstruct(a)
    ref=np.load(ROOT/'notebooks/reference_outputs/upstream_reconstruction.npz')['reconstruction'][0]
    np.testing.assert_allclose(out,ref,rtol=5e-5,atol=5e-5)
