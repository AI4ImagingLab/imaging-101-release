from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import modified_laplacian, initial_focal_coordinate, polynomial_operator, fit_focus_polynomials

FIX=Path(__file__).resolve().parents[1]/'fixtures'

def test_native_normal_equations_and_pseudoinverse():
    with np.load(FIX/'upstream_stages.npz',allow_pickle=False) as f:
        X,P=polynomial_operator(19,8)
        np.testing.assert_array_equal(X,f['vandermonde'])
        A=(X.astype(float).T@X.astype(float)).astype(np.float32)
        np.testing.assert_allclose(A,f['normal_matrix'],rtol=2e-7,atol=0)
        np.testing.assert_allclose(P,f['pseudoinverse'],atol=2e-6,rtol=0)

def test_native_focus_coefficients_and_derivatives():
    with np.load(FIX/'upstream_stages.npz',allow_pickle=False) as f:
        sharp=f['raw_sharpness'][:,:,None]
        c,d=fit_focus_polynomials(sharp)
        np.testing.assert_allclose(c[:,:,0],f['positive_focus_coefficients'],atol=2e-6,rtol=0)
        np.testing.assert_allclose(d[:,:,0],f['negative_focus_derivative'],atol=1e-6,rtol=0)

@pytest.mark.parametrize('k',range(6))
def test_native_initializer_with_original_image_halo(k):
    with np.load(FIX/f'native_window_{k}.npz',allow_pickle=False) as f:
        sharp=modified_laplacian(f['observations'])
        actual=initial_focal_coordinate(sharp)
        np.testing.assert_allclose(actual[18:-18,18:-18],f['initial'][18:-18,18:-18],atol=1e-4,rtol=0)
