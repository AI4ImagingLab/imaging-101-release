from pathlib import Path
import importlib
import numpy as np
from src.main import run
from src.solvers import reconstruct,INITIAL_VALUES,FREE_INDICES
from src.physics_model import physical_jacobian
from src.visualization import observation_metrics

ROOT=Path(__file__).resolve().parents[2]


def test_full_original_lm_then_trr_parity_and_identifiability():
    with np.load(ROOT/'test/fixtures/upstream_numeric.npz',allow_pickle=False) as ref:
        fresh=reconstruct(ref['input_fid'])
        np.testing.assert_allclose(fresh['first_parameters'].ravel(),ref['output_first_parameters'],rtol=1e-10,atol=1e-9)
        # Convergedphysical outputs are tested against criteria frozen from three
        # independentoriginalAPIstarts before changing localprecision controls.
        p=ref['output_final_parameters'].reshape(16,5);f=fresh['parameters']
        assert np.sqrt(np.mean((f[:,0]-p[:,0])**2))/np.ptp(p[:,0])<1e-6
        assert np.max(abs(f[:,1]-p[:,1]))<.001
        assert np.max(abs(f[:,2]-p[:,2]))/np.pi<.001
        assert abs(np.angle(np.exp(1j*(f[0,3]-p[0,3]))))<1e-5
        modeled=fresh['fitted_fid'];upstream=ref['output_final_fid'][::2]+1j*ref['output_final_fid'][1::2]
        np.testing.assert_allclose(modeled,upstream,rtol=1e-6,atol=1e-7)
        # Operatorparity uses exactlythe same originalparameters, separating
        # derivative correctness from convergedoptimizer's finite precision.
        np.testing.assert_allclose(physical_jacobian(p,ref['input_time']),ref['output_final_jacobian'],rtol=1e-9,atol=1e-9)
        assert fresh['parameters'].shape==(16,5) and len(fresh['free_values'])==37
        assert fresh['objective']<115
        assert observation_metrics(ref['input_fid'],modeled)['relative_l2']<.19
        # Physical free-parameter Jacobian from independent central differences
        # has rank37; covariance/line-shape gauges are constrained, not hidden.
        from src.solvers import residual
        theta=fresh['free_values'];columns=[]
        for k in range(37):
            delta=1e-5*max(1,abs(theta[k]));a=theta.copy();b=theta.copy();a[k]+=delta;b[k]-=delta
            columns.append((residual(a,ref['input_time'],ref['input_fid'])-residual(b,ref['input_time'],ref['input_fid']))/(2*delta))
        assert np.linalg.matrix_rank(np.column_stack(columns))==37


def test_independent_public_prior_converges_to_same_estimand():
    with np.load(ROOT/'data/raw_data.npz') as z:fid=z['measurements'][0]
    first=reconstruct(fid);second=reconstruct(fid,use_lm=False)
    assert np.sqrt(np.mean((first['reconstruction']-second['reconstruction'])**2))/np.ptp(first['reconstruction'])<1e-6
    assert np.max(abs(first['frequency_hz']-second['frequency_hz']))<.001
    assert np.max(abs(first['linewidth_hz']-second['linewidth_hz']))<.001
    assert abs(first['objective']-second['objective'])<1e-9


def test_entrypoint_raw_exactness_output_shapes_and_backend(tmp_path):
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as raw,np.load(ROOT/'test/fixtures/initial_stage.npz',allow_pickle=False) as fixture:
        np.testing.assert_array_equal(raw['measurements'][0],fixture['input_fid'])
        assert raw['measurements'].dtype==np.complex128
    import matplotlib
    backend=matplotlib.get_backend();importlib.reload(importlib.import_module('src.main'));assert matplotlib.get_backend()==backend
    p=tmp_path/'result.npz';run(output_path=p)
    with np.load(p,allow_pickle=False) as prediction:
        assert set(prediction.files)=={'reconstruction','frequency_hz','linewidth_hz','phase_rad'}
        for key in prediction.files:
            assert prediction[key].dtype==np.float64 and np.isfinite(prediction[key]).all()
            assert prediction[key].shape==((1,1) if key=='phase_rad' else (1,16))
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['10000','120','300','1024','16','37']:assert value in readme
