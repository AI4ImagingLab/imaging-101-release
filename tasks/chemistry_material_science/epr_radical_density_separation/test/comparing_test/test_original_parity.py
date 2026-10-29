import numpy as np

def relative(a,b):return np.linalg.norm(a-b)/np.linalg.norm(b)

def test_direct_fourier_forward_and_adjoint(model,original):
 assert relative(model.project(original['operator_test_density']),original['operator_test_forward'])<1e-9
 assert relative(model.adjoint(original['operator_test_observations']),original['operator_test_adjoint'])<1e-9

def test_toeplitz_kernels_and_normal(model,original):
 from src.physics_model import NormalOperator
 kernels=model.kernels();assert relative(kernels,original['toeplitz_kernels'])<1e-9
 assert relative(NormalOperator(kernels,model.shape)(original['operator_test_density']),original['operator_test_normal'])<1e-9

def test_full_original_updates_and_field(model,public,original):
 from src.solvers import reconstruct
 u,d=reconstruct(public['projections'],model)
 for key in ('initialization','backprojected_observations','lipschitz','unnormalized_tv_weight'):assert relative(d[key],original[key])<1e-9
 for key in ('first_iterate_inputs','first_iterate_gradients'):assert relative(d[key],original[key])<1e-8
 assert relative(u,original['original_reconstruction'])<1e-7

def test_readonly_energy_tracking_is_identical(model,public):
 from src.solvers import reconstruct
 a,_=reconstruct(public['projections'],model);b,d=reconstruct(public['projections'],model,track_energy=True)
 assert np.array_equal(a,b);assert len(d['energy_history'])>10 and np.isfinite(d['energy_history']).all()

def test_point_source_independent_shift(model,original):
 u=np.zeros((2,100,100));u[0]=original['point_density']
 assert relative(model.project(u),original['point_forward_independent'])<1e-10

def test_judge_independent_forward(judge,public,original):
 y=judge.predict_observations(original['operator_test_density'],public['magnetic_field_G'],public['gradients_G_cm'],public['reference_spectra'])
 assert relative(y,original['operator_test_forward'])<1e-9
