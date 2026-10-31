from pathlib import Path
import numpy as np
from src.physics_model import thermal_properties,laplacian,SPACING,TIME_STEP
from src.preprocessing import load_data
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_exact_bound_real_measurements():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_hfits.npz') as a:
        np.testing.assert_array_equal(data['temperature'],a['input_temperature_kelvin'])
        assert a['param_dx_m']==SPACING and a['param_dt_s']==TIME_STEP
        assert np.isfinite(data['temperature']).all()

def test_upstream_material_properties_and_conduction_stage():
    data=load_data(ROOT/'data/raw_data.npz');k,cp,rho=thermal_properties(np.transpose(data['temperature'],(1,2,0)))
    with np.load(ROOT/'test/fixtures/upstream_hfits.npz') as a:
        np.testing.assert_allclose(k,a['output_k'],rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(cp,a['output_cp'],rtol=1e-14,atol=1e-12)
        assert rho==a['output_rho']
        field=data['temperature'][10];x=.0127;y=.0127;i=j=20
        expected=np.array([((field[i-1,j]-field[i,j])+(field[i+1,j]-field[i,j]))*y/x,((field[i,j-1]-field[i,j])+(field[i,j+1]-field[i,j]))*x/y])
        expected*=thermal_properties(field[i,j])[0]*.000793
        np.testing.assert_allclose(expected,a['output_conductive_gradient'],rtol=1e-13,atol=1e-13)

def test_full_original_pipeline_parity():
    q,terms=reconstruct(load_data(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_hfits.npz') as a:
        np.testing.assert_allclose(q,a['output_incident_radiation'],rtol=2e-12,atol=2e-12)
    assert q.shape==(25,64,48) and np.isfinite(q).all()
