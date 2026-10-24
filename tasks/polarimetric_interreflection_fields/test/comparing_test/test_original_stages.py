from pathlib import Path
import numpy as np
from src.physics_model import design_matrix,components
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_design_and_pseudoinverse():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        W=design_matrix(z['output_polarizer_angles_psg'],z['output_polarizer_angles_psa'])
        np.testing.assert_allclose(W,z['output_W'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(np.linalg.pinv(W),z['output_W_pinv'],rtol=1e-10,atol=1e-12)

def test_actual_original_nonconstant_crop_coefficients_and_all_five_quantities():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for i in (0,1):
            raw=np.moveaxis(z[f'input_crop{i}_intensities'],-1,0)
            r=reconstruct(raw,z['output_polarizer_angles_psg'],z['output_polarizer_angles_psa'])
            np.testing.assert_allclose(r['coefficients'],z[f'output_crop{i}_X'],rtol=1e-10,atol=1e-12)
            comp=components(r['reconstruction'])
            for name,original in [('sum_amplitude','intensity_forward'),('difference_amplitude','intensity_reverse'),('unpolarized','intensity_unpolarized')]:
                np.testing.assert_allclose(comp[name],z[f'original_{original}_crop{i}'],rtol=1e-10,atol=1e-12)
            for name,original in [('sum_phase','phase_forward'),('difference_phase','phase_reverse')]:
                np.testing.assert_allclose(np.exp(2j*comp[name]),np.exp(2j*z[f'original_{original}_crop{i}']),rtol=1e-10,atol=1e-12)
