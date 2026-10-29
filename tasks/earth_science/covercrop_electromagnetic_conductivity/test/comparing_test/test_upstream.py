import numpy as np
from src.preprocessing import load_data
from src.physics_model import jacobian,forward
from src.solvers import reconstruct
def test_frozen_upstream_stages_and_full_output():
    d=load_data()
    with np.load('test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(d['apparent_conductivity'],f['input_observations'])
        np.testing.assert_array_equal(d['station_coordinates'],f['param_positions'])
        np.testing.assert_allclose(jacobian(d['layer_interfaces'],d['coil_spacing']),f['output_jacobian'],rtol=2e-14,atol=1e-15)
        rec,diag=reconstruct(d);assert all(x['success'] for x in diag)
        np.testing.assert_allclose(rec,f['output_reconstruction'],rtol=2e-6,atol=1e-5)
        prediction=np.array([forward(v,d['layer_interfaces'],d['coil_spacing']) for v in rec])
        np.testing.assert_allclose(prediction,f['output_prediction'],rtol=2e-6,atol=1e-5)
def test_entrypoint_and_quality():
    from src.main import main
    main()
    with np.load('output/reconstruction.npz',allow_pickle=False) as a:assert a['reconstruction'].shape==(1,30,7)
