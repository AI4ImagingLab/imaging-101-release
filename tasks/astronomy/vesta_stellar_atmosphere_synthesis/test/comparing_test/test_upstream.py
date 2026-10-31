from pathlib import Path
import numpy as np,pytest
from src.preprocessing import load_observations,mask_flux
from src.physics_model import SpectralModel,build_operators
from src.solvers import prepare_likelihood,estimate_error_map,stretch_posterior,initialize_walkers,LOWER,UPPER

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def model():return SpectralModel(load_observations(ROOT/'data/raw_data.npz'))

def test_all_original_physical_forward_cases(model):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as historical,np.load(ROOT/'test/fixtures/upstream_sensitive_stages.npz',allow_pickle=False) as current:
        for i in range(4):
            p=current[f'parameters_{i}']
            np.testing.assert_allclose(model.forward_fidelity(p),current[f'forward_{i}'],rtol=1e-9,atol=1e-6)
            np.testing.assert_allclose(model.forward(p),current[f'forward_{i}'],rtol=1e-9,atol=1e-6)
            np.testing.assert_allclose(model.forward(p,masked=True),current[f'masked_forward_{i}'],rtol=1e-9,atol=1e-6)
            assert np.all(model.forward(p,masked=True)>0)
            # Different backend's independent source acquisition remains labeled.
            np.testing.assert_allclose(model.forward(historical[f'parameters_{i}']),historical[f'forward_{i}'],rtol=1e-9,atol=1e-6)
        points=np.stack([current[f'parameters_{i}'] for i in range(4)])
        batch=model.forward_batch(points)
        for i,p in enumerate(points):np.testing.assert_allclose(batch[i],model.forward(p),rtol=1e-10,atol=1e-6)

def test_source_order_finite_difference_jacobians(model):
    with np.load(ROOT/'test/fixtures/upstream_finite_difference.npz',allow_pickle=False) as z:
        actual=np.stack([model.forward_fidelity(p,masked=True) for p in z['parameters']])
        for i,p in enumerate(z['parameters']):np.testing.assert_array_equal(model.interpolate_weights_fidelity(p),z['weights'][i])
        np.testing.assert_array_equal(actual,z['spectra'])
        for multiplier,start in [(1.,1),(4.,5)]:
            measured=(actual[start:start+4]-actual[0])/z['h'][:,None]/multiplier
            original=(z['spectra'][start:start+4]-z['spectra'][0])/z['h'][:,None]/multiplier
            np.testing.assert_array_equal(measured,original)

def test_independent_static_operator_rebuild_without_observed_flux(model):
    raw=model.raw
    known={key:raw[key] for key in ['wavelength','model_wavelength','pca_basis','pca_mean','pca_scale']}
    known['ranges']=raw['ranges'][:1]
    rebuilt=build_operators(known)
    v=int(raw['operator_velocity_offsets'][1]);o=int(raw['operator_observation_offsets'][1]);p=int(raw['operator_post_offsets'][1])
    limits=dict(operator_pre_basis=(slice(None),slice(0,v)),operator_pre_mean=slice(0,v),operator_velocity_xk=slice(0,v),operator_wmid=slice(0,1),operator_post_values=slice(0,p),operator_gaussian_offset=slice(0,o),operator_velocity_offsets=slice(0,2),operator_observation_offsets=slice(0,2),operator_post_offsets=slice(0,2),operator_observation_indices=slice(0,o))
    for key,selection in limits.items():np.testing.assert_allclose(rebuilt[key],raw[key][selection],rtol=1e-10,atol=1e-10)

def test_complete_original_sensitive_pipeline(model):
    likelihood=prepare_likelihood(model)
    with np.load(ROOT/'test/fixtures/upstream_sensitive_stages.npz',allow_pickle=False) as z:
        for i,tolerance in enumerate([1e-4,1e-4,1e-4,1e-3]):
            np.testing.assert_allclose(likelihood['first_parameters'][i],z['do_curve_fit_0_0'][i],rtol=0,atol=tolerance)
            np.testing.assert_allclose(likelihood['parameters'][i],z['param_result'][i],rtol=0,atol=tolerance)
        for actual,expected in [('covariance','cov_par'),('first_covariance','do_curve_fit_0_1'),('parameter_error','epar'),('walker_width','svec'),('first_factor','do_curve_fit_0_2'),('second_factor','do_curve_fit_1_2'),('errors','err_tot'),('masked_input_error','masked_eflux_obs'),('full_error_map','extra_err_rank'),('masked_error_map','extra_err')]:np.testing.assert_allclose(likelihood[actual],z[expected],rtol=1e-9,atol=1e-6)
        np.testing.assert_array_equal(LOWER,z['lower_bounds']);np.testing.assert_array_equal(UPPER,z['upper_bounds'])
        assert np.all(z['sigma_prior']<=0)
        map_details=likelihood['error_map_details']
        for i,result in enumerate(map_details['per_window']):
            for j,value in enumerate(result):np.testing.assert_allclose(value,z[f'get_error_{i}_{j}'],rtol=1e-9,atol=1e-6)
        for i,key in enumerate(['centers','widths','amplitudes','scatter']):np.testing.assert_allclose(map_details[key],z[f'wide_errors_0_{i}'],rtol=1e-9,atol=1e-6)
        for i,value in enumerate(mask_flux(model.raw['wavelength'],model.raw['flux'],likelihood['full_error_map'],model.raw['masks'])):np.testing.assert_allclose(value,z[f'mask_flux_1_{i}'],rtol=1e-9,atol=1e-6)
        np.testing.assert_allclose(likelihood['full_error_map'],z['do_error_map_0_0'],rtol=1e-9,atol=1e-6)
        np.testing.assert_allclose(likelihood['masked_error_map'],z['do_error_map_0_1'],rtol=1e-9,atol=1e-6)
        # Remaining identified stage aliases and immutable acquisition constants.
        for key,values in [('wave_obs_rank',model.raw['wavelength']),('flux_obs_rank',model.raw['flux']),('eflux_obs_rank',model.raw['uncertainty']*likelihood['second_factor']),('central_wave_rank',model.raw['ranges'].mean(1)),('width_wave_rank',np.diff(model.raw['ranges'],axis=1)[:,0]/2),('rank_order',model.raw['continuum_order']),('central_wave_mask',model.raw['masks'][:,0]),('width_wave_mask',model.raw['masks'][:,1]),('masked_wave_obs',likelihood['masked_wavelength']),('masked_flux_obs',likelihood['masked_flux'])]:np.testing.assert_allclose(values,z[key],rtol=1e-9,atol=1e-6)
        np.testing.assert_array_equal(likelihood['masked_velocity'],z['mask_flux_0_3'])
        np.testing.assert_array_equal(likelihood['ranked_velocity'],z['mask_flux_0_4'])

def test_short_original_sampler_statistics_not_seed_sample_identity(model):
    with np.load(ROOT/'test/fixtures/upstream_sensitive_stages.npz',allow_pickle=False) as z:
        likelihood=dict(parameters=z['param_result'],walker_width=z['svec'],errors=z['err_tot'],masked_flux=z['masked_flux_obs'])
        positions=initialize_walkers(likelihood['parameters'],likelihood['walker_width'],model,likelihood)
        reference=z['initial_positions'];assert positions.shape==(16,4) and np.isfinite(positions).all()
        initial_scale=np.std(reference,axis=0,ddof=1)
        assert np.all(abs(positions.mean(0)-reference.mean(0))<=3*initial_scale/np.sqrt(16))
        chain,logp,acceptance=stretch_posterior(model,likelihood,burn=0,production=2)
        reference=z['first_two_chain'];assert chain.shape==reference.shape and np.isfinite(chain).all() and np.isfinite(logp).all()
        sample=chain.reshape(-1,4);original=reference.reshape(-1,4);scale=np.std(original,axis=0,ddof=1)
        assert np.all(abs(sample.mean(0)-original.mean(0))<=3*scale/np.sqrt(len(sample)))
        ratio=np.var(sample,axis=0)/np.var(original,axis=0);assert np.all((ratio>.5)&(ratio<2))
        assert abs(logp.mean()-z['first_two_log_probability'].mean())<=3*z['first_two_log_probability'].std()/np.sqrt(logp.size)
        assert np.all((acceptance>=0)&(acceptance<=1))
