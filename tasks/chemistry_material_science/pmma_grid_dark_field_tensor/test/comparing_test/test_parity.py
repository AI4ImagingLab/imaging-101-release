from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
from src.physics_model import moments_from_coefficients
ROOT=Path(__file__).resolve().parents[2]
def test_original_closed_form_tensor():
    z=np.load(ROOT/'test/fixtures/upstream_coefficients.npz');c,t,r=moments_from_coefficients(z['output_grid_coefficients'],z['output_sample_coefficients'])
    np.testing.assert_allclose(c,z['output_covariance'],rtol=1e-10,atol=1e-10);np.testing.assert_array_equal(t,z['output_transmission']);np.testing.assert_array_equal(r,z['output_ratios'])
def test_original_fits_and_complete_pipeline():
    grid,sample=load_data(ROOT/'data/raw_data.npz');c,t,diagnostics=reconstruct(grid,sample);z=np.load(ROOT/'test/fixtures/upstream_coefficients.npz')
    for actual,key in zip(diagnostics['coefficients'],['output_grid_coefficients','output_sample_coefficients']):np.testing.assert_allclose(actual,z[key],rtol=1e-8,atol=1e-8,equal_nan=True)
    np.testing.assert_allclose(c,z['output_covariance'],rtol=1e-6,atol=1e-6);np.testing.assert_allclose(t,z['output_transmission'],rtol=1e-8,atol=1e-8)
