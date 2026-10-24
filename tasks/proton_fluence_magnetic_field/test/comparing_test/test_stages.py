from pathlib import Path
import numpy as np
import pytest
from src.solvers import reconstruct
from src.physics_model import velocity_from_mapping, magnetic_from_mapping
from src.preprocessing import load_observations, uniform_incident_fluence

FIX = Path(__file__).resolve().parents[1] / 'fixtures'

@pytest.mark.parametrize('steps', [1, 2, 7])
def test_original_full_grid_steps(steps):
    with np.load(FIX / 'upstream_stages.npz', allow_pickle=False) as f:
        result = reconstruct(f['input_x'], f['input_y'], f['input_flux'], max_steps=steps)
        for name in ['phin', 'phix', 'phiy', 'phixx', 'phiyy', 'phixy', 'Fn', 'fluxn', 'det', 'RMSfluxerror']:
            expected = f[f'output_native_{steps}_{name}']
            np.testing.assert_allclose(result[name], expected, rtol=1e-10,
                                       atol=1e-12 * max(np.max(np.abs(expected)), 1e-20))

def test_original_rectangular_nonzero_mixed_hessian():
    with np.load(FIX / 'upstream_stages.npz', allow_pickle=False) as f:
        x, y = f['input_crop_x'], f['input_crop_y']
        assert np.array_equal(uniform_incident_fluence(f['input_crop_flux']),f['input_crop_flux0'])
        result = reconstruct(x, y, f['input_crop_flux'], max_steps=7)
        assert np.max(np.abs(f['output_crop_7_phixy'])) > .009
        for name in ['phin', 'phix', 'phiy', 'phixx', 'phiyy', 'phixy', 'Fn', 'fluxn', 'det']:
            expected = f[f'output_crop_7_{name}']
            np.testing.assert_allclose(result[name], expected, rtol=1e-10, atol=1e-12*max(np.max(np.abs(expected)),1e-20))
        np.testing.assert_allclose(velocity_from_mapping(x,y,result['phix'],result['phiy']),f['output_crop_velocity'],rtol=1e-10,atol=1e-5)
        np.testing.assert_allclose(result['reconstruction'],f['output_crop_magnetic'],rtol=1e-10,atol=1e-10)

def test_bound_observations_are_frozen_source_arrays():
    data = load_observations()
    with np.load(FIX / 'upstream_stages.npz', allow_pickle=False) as f:
        for a,b in [('x','input_x'),('y','input_y'),('fluence','input_flux')]:
            assert np.array_equal(data[a],f[b])
        assert np.array_equal(uniform_incident_fluence(data['fluence']),f['input_flux0'])
