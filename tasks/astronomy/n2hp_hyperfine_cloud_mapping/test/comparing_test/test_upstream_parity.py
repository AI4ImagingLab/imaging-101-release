from pathlib import Path
import numpy as np
from src.physics_model import brightness,optical_depth,predict_cube
from src.preprocessing import load_observations,estimate_noise
from src.solvers import reconstruct,OUTPUT_KEYS
ROOT=Path(__file__).resolve().parents[2]

def test_forward_and_opacity_independent_upstream_fixture():
    with np.load(ROOT/'test/fixtures/upstream_run.npz',allow_pickle=False) as u:
        _,f=load_observations(ROOT/'data/raw_data.npz')
        np.testing.assert_allclose(brightness(f,u['probe_parameters']),u['forward'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(optical_depth(f,u['probe_parameters']),u['tau'],rtol=1e-11,atol=1e-11)

def test_preprocessing_matches_original_radio_axis_noise():
    cube,f=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_run.npz',allow_pickle=False) as u:
        np.testing.assert_allclose(estimate_noise(cube,f),u['noise'],rtol=0,atol=0)

def test_all_nine_fits_and_weighted_objective_match_upstream():
    cube,f=load_observations(ROOT/'data/raw_data.npz');fresh,chi=reconstruct(cube,f)
    p=np.stack([fresh[k][0] for k in OUTPUT_KEYS]);model=predict_cube(f,p)
    with np.load(ROOT/'test/fixtures/upstream_run.npz',allow_pickle=False) as u:
        for i in range(4):assert np.max(abs((p[i]-u['parameters'][0,i])/u['parameters'][0,i]))<=2e-4
        assert np.sqrt(np.mean((model-u['model'][0])**2))<=1e-5
        np.testing.assert_allclose(chi,u['chi_square'],rtol=1e-7,atol=1e-5)
        residual=(model-cube)/u['noise'][None]
        np.testing.assert_allclose(chi,(residual**2).sum(axis=0),rtol=1e-14)


def test_local_forward_numerical_jacobian_matches_original_model():
    _,f=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_jacobian.npz',allow_pickle=False) as u:
        p=u['probe_parameters'];columns=[]
        for i in range(4):
            step=p[i]*float(u['relative_step']);a=p.copy();b=p.copy();a[i]+=step;b[i]-=step
            columns.append((brightness(f,a)-brightness(f,b))/(2*step))
        np.testing.assert_allclose(np.array(columns).T,u['jacobian'],rtol=2e-8,atol=1e-8)
