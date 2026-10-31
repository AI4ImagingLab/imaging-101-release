from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.physics_model import modal_model,augmented_model,matern52,hyperparameter_cost
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_operators_kernel_and_full_hyperparameter_objective():
    data=load_observations(ROOT/'data/raw_data.npz');m=modal_model(data['mass'],data['stiffness'],data['damping'],data['sensor_matrix'],.01)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        sign=np.sign(np.diag(m['Phi_r'].T@data['mass']@z['model_Phi_r']));D=np.diag(np.r_[sign,sign,np.repeat(sign,3)])
        np.testing.assert_allclose(m['Phi_r']*sign,z['model_Phi_r'],atol=1e-14)
        for key in ['A_c','B_c','Omega2_r','Gamma_r']:
            np.testing.assert_allclose(m[key],z['model_'+key],atol=1e-12,rtol=1e-10)
        aug=augmented_model(m,z['output_hyperparameters']);knl=matern52(z['output_hyperparameters'],.01)
        for key in ['F_ac','Q_ac','P_0','F_ad','Q_ad']:
            np.testing.assert_allclose(aug[key],D@z['augmented_'+key]@D,atol=1e-9,rtol=1e-8)
        for key in ['H_ad','H_ae']:np.testing.assert_allclose(aug[key]@D,z['augmented_'+key],atol=1e-12,rtol=1e-10)
        for key in ['F_c','L_c','H_c','F','L','H','P','Q']:np.testing.assert_allclose(knl[key],z['kernel_'+key],atol=1e-10,rtol=1e-8)
        costs=[hyperparameter_cost(x,data['accelerations'],m) for x in z['input_hyperparameter_cases'].T]
        np.testing.assert_allclose(costs,z['output_hellinger_costs'],atol=1e-12,rtol=1e-8)

def test_full_fixed_hp_kalman_smoother_and_confidence_stages():
    data=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        fields,info=reconstruct(data,hyperparameters=z['output_hyperparameters'])
        sign=np.sign(np.diag(info['model']['Phi_r'].T@data['mass']@z['model_Phi_r']));D=np.diag(np.r_[sign,sign,np.repeat(sign,3)])
        np.testing.assert_allclose(info['state']@D,z['output_state'].T,atol=1e-6,rtol=1e-6)
        for name in ['first_filtered_state','first_predicted_state']:
            np.testing.assert_allclose(info[name]@D,z['output_'+name].T,atol=1e-7,rtol=1e-6)
        for name in ['first_filtered_covariance','first_predicted_covariance']:
            np.testing.assert_allclose(D@info[name]@D,z['output_'+name].transpose(2,0,1),atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(D@info['covariance_samples']@D,z['output_covariance_samples'].transpose(2,0,1),atol=1e-6,rtol=1e-6)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_measurements.npz',allow_pickle=False) as z:
        for key in ['reconstruction','velocity','acceleration']:
            reference=z[key][0];error=np.sqrt(np.mean((fields[key]-reference)**2))/np.ptp(reference);assert error<1e-6
        for key,native in [('reconstruction','displacement_variance'),('velocity','velocity_variance'),('acceleration','acceleration_variance')]:
            np.testing.assert_allclose(info['variance'][key],z[native][0],atol=1e-7,rtol=1e-5)

def test_complete_relearned_hp_pipeline():
    fields,info=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'notebooks/reference_outputs/upstream_measurements.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(info['hyperparameters'],z['hyperparameters'][0],rtol=1e-3)
        for key in ['reconstruction','velocity','acceleration']:
            ref=z[key][0];assert np.sqrt(np.mean((fields[key]-ref)**2))/np.ptp(ref)<1e-4
            assert np.sum(fields[key]*ref)/np.linalg.norm(fields[key])/np.linalg.norm(ref)>.99999
