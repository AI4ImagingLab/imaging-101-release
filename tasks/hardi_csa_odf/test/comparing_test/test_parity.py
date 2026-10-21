from pathlib import Path
import numpy as np
from src.preprocessing import load_data, normalize
from src.physics_model import fit_operator
from src.solvers import reconstruct

ROOT = Path(__file__).resolve().parents[2]


def test_original_stages_and_pipeline():
    data = load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as expected:
        b = data['bvalues'][0]; g = data['directions'][0]
        B, A = fit_operator(g[b > 50])
        np.testing.assert_allclose(B, expected['basis'], atol=2e-7, rtol=0)
        np.testing.assert_allclose(A, expected['fit_matrix'], atol=2e-7, rtol=0)
        np.testing.assert_array_equal(normalize(data['signal'], b)[0], expected['normalized'])
        result = reconstruct(data['signal'], b, g)
        assert np.linalg.norm(result-expected['reconstruction'])/np.linalg.norm(expected['reconstruction']) < 1e-6


def test_standard_entry_point():
    from src.main import main
    result = main()
    with np.load(ROOT/'output/reconstruction.npz') as actual:
        np.testing.assert_array_equal(actual['reconstruction'], result)


def test_archived_notebook_measurement_consistency():
    data = load_data(ROOT/'data/raw_data.npz')
    result = reconstruct(data['signal'], data['bvalues'][0], data['directions'][0])
    with np.load(ROOT/'notebooks/reference_outputs/cleaned.npz') as measurement:
        np.testing.assert_allclose(result, measurement['reconstruction'], atol=1e-13, rtol=0)
