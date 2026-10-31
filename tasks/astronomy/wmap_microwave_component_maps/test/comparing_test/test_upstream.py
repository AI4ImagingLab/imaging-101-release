import numpy as np
from src.preprocessing import load_data
from src.physics_model import mixing_matrix
from src.solvers import reconstruct
def test_frozen_author_forward_inverse_and_covariance():
    d=load_data()
    with np.load('test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(d['measurements'],z['input_measurements'])
        np.testing.assert_array_equal(d['frequencies'],z['param_frequencies'])
        np.testing.assert_allclose(mixing_matrix(d['frequencies']),z['output_mixing'],rtol=2e-14,atol=2e-15)
        m,p,c=reconstruct(d)
        np.testing.assert_allclose(m,z['output_reconstruction'],rtol=1e-11,atol=2e-10)
        np.testing.assert_allclose(p,z['output_prediction'],rtol=1e-11,atol=2e-10)
        np.testing.assert_allclose(c,z['output_inverse_covariance'],rtol=1e-12,atol=1e-12)
def test_cli_all_required_outputs():
    from src.main import main
    main()
    with np.load('output/reconstruction.npz',allow_pickle=False) as z:
        assert set(z.files)=={'reconstruction','synchrotron','dust'}
        assert all(z[k].shape==(1,48,64) for k in z.files)
