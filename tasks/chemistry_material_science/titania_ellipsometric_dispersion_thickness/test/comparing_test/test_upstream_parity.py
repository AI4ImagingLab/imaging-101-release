from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,measured_rho
from src.physics_model import predict_rho,cauchy_index
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_complete_observations_and_calibration_against_prelocal_source():
    data=load_observations()
    with np.load(ROOT/'test/fixtures/output_upstream_fit.npz',allow_pickle=False) as z:
        for key,original in [('wavelength_nm','wavelength_nm'),('psi_deg','psi_degree'),
                             ('delta_deg','delta_degree'),('silicon_index','silicon_index')]:
            np.testing.assert_array_equal(data[key],z[original])
        np.testing.assert_allclose(measured_rho(data),z['rho_observed'],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(predict_rho(data,z['parameters']),z['rho_fitted'],rtol=1e-12,atol=1e-12)

def test_full_cleaned_inverse_all_four_fields_against_prelocal_source():
    output,record=reconstruct(load_observations())
    with np.load(ROOT/'test/fixtures/output_upstream_fit.npz',allow_pickle=False) as z:
        errors=abs(output['parameters'][0]-z['parameters'])
        assert np.all(errors<=np.array([1e-7,5e-4,1e-3,1e-7]))
        from src.visualization import rho_angles
        psi,delta=rho_angles(z['rho_fitted'])
        np.testing.assert_allclose(psi,z['psi_fitted'],atol=1e-11)
        np.testing.assert_allclose((delta-z['delta_fitted']+180)%360-180,0.,atol=1e-11)
        np.testing.assert_allclose(np.sqrt(np.diag(record['covariance'])),z['parameter_stderr'],rtol=1e-4,atol=1e-8)
        reference=cauchy_index(z['wavelength_nm'],z['parameters'])
        assert np.sqrt(np.mean((output['reconstruction'][0]-reference)**2))/np.ptp(reference)<1e-7
    assert record['nfev']==32 and record['success']

def test_documented_angle_discrepancy_is_a_real_measured_sensitivity():
    data=load_observations()
    with np.load(ROOT/'test/fixtures/output_upstream_fit.npz',allow_pickle=False) as z:
        parameters=z['parameters'];main=predict_rho(data,parameters);actual=predict_rho(data,parameters,70.06)
    assert .005<np.linalg.norm(actual-main)/np.linalg.norm(main)<.006

def test_every_prelocal_physics_fixture_has_a_numerical_consumer():
    from src.physics_model import reflection
    data=load_observations()
    with np.load(ROOT/'test/fixtures/output_source_physics.npz',allow_pickle=False) as z:
        p=z['source_parameters'];rho=predict_rho(data,p)
        np.testing.assert_allclose(rho,z['independent_rho'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(predict_rho(data,p,70.06),z['rho_actual_angle'],rtol=1e-12,atol=1e-12)
        rs,rp=reflection(data['wavelength_nm'],p,data['silicon_index'])
        np.testing.assert_allclose(abs(rs)**2,z['reflectance_s'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(abs(rp)**2,z['reflectance_p'],rtol=1e-12,atol=1e-12)
        scale=np.maximum(abs(p),1.);columns=[]
        for j in range(4):
            step=1e-5*scale[j];plus=p.copy();minus=p.copy();plus[j]+=step;minus[j]-=step
            change=(predict_rho(data,plus)-predict_rho(data,minus))/(2*step)*scale[j]
            columns.append(np.r_[change.real,change.imag])
        jac=np.array(columns).T
        np.testing.assert_allclose(jac,z['scaled_jacobian'],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(np.linalg.svd(jac,compute_uv=False),z['singular_values'],rtol=1e-7,atol=1e-8)

