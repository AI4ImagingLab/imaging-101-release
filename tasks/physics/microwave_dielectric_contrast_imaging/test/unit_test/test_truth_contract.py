from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def test_actual_fine_truth_exact_physical_cell_area_mean():
    with np.load(ROOT/'test/fixtures/truth_averaging_case.npz',allow_pickle=False) as f:native=f['input_native_contrast']
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as f:coarse=f['reconstruction']
    np.testing.assert_array_equal(native.reshape(1,30,2,30,2).mean(axis=(2,4)),coarse)
