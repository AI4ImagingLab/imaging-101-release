from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import TensorAcquisition,render
from src.solvers import reconstruct
from src.cuda_runtime import project
ROOT=Path(__file__).resolve().parents[2]

def test_original_complete_impulse_geometry():
    z=np.load(ROOT/'test/fixtures/upstream_impulses.npz');shape=tuple(z['param_shape']);images=np.eye(np.prod(shape),dtype='float32').reshape(-1,*shape)
    actual=project(images,z['param_angles'],int(z['param_rays'])).reshape(len(images),-1).T
    np.testing.assert_array_equal(actual,z['output_gpu_matrix'])

def test_original_all_stages_and_full_pipeline():
    data=load_data(ROOT/'data/raw_data.npz');field,diagnostics=reconstruct(data);model=diagnostics['model'];z=np.load(ROOT/'test/fixtures/upstream_stages.npz');probe=z['input_probe']
    np.testing.assert_array_equal(render(model.basis,probe,model.shape),z['output_basis'])
    np.testing.assert_array_equal(model.mean_project(z['output_basis']),z['output_projection'])
    np.testing.assert_array_equal(model.forward(probe),z['output_forward'])
    np.testing.assert_array_equal(np.sum(model.sampler*model.mean_project(z['output_basis']),axis=0),z['output_sample'])
    np.testing.assert_array_equal((model.jacobian@probe.ravel()).reshape(500,192),z['output_jacobian_probe'])
    np.testing.assert_array_equal(model.weights,z['output_weights'])
    np.testing.assert_array_equal(model.inverse[model.mask],z['output_ray_norm'])
    np.testing.assert_array_equal(model.mask,z['output_sinomask'])
    np.testing.assert_array_equal(model.adjoint(model.weights*data['strain_measurements']),z['output_backward_corrected'])
    defect=np.linalg.norm(z['output_backward']-z['output_backward_corrected'])/np.linalg.norm(z['output_backward_corrected'])
    assert defect>1e-3  # Explicitly retain evidence of the named source correction.
    with np.load(ROOT/'test/fixtures/upstream_projected_basis.npz') as p:np.testing.assert_array_equal(model.projected_basis,p['output_projected_basis'])
    with np.load(ROOT/'notebooks/reference_outputs/corrected_oracle.npz') as original:np.testing.assert_array_equal(field,original['output_reconstruction']);np.testing.assert_array_equal(diagnostics['coefficients'],original['output_coefficients'])
    assert diagnostics['iterations']==108 and diagnostics['success']
