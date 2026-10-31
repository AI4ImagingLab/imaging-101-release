from pathlib import Path
import numpy as np
from src.solvers import reconstruct,bas_reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_frozen_original_small_cases():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as z:
        xy=np.stack([z['input_small_x'],z['input_small_y']],axis=-1);v=z['input_small_voltage']
        np.testing.assert_allclose(reconstruct(xy,v),z['output_small_geiser'],rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(bas_reconstruct(xy,v),z['output_small_bas'],rtol=2e-13,atol=2e-13)
