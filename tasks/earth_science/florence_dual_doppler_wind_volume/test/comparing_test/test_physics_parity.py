from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data,normalize_observations
from src.physics_model import fall_speed,derivative,derivative_adjoint,objective_gradient,divergence,forward
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def data():return load_data(ROOT/'data/raw_data.npz')
def test_original_preprocessing_terminal():
    d=data()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        assert_allclose(normalize_observations(d),f['param_rmsVr'],rtol=1e-14)
        mask=d['validity']>0;assert_allclose(fall_speed(d['reflectivity'],d['z'])[mask],f['output_fall_speed'][mask],rtol=5e-14)
def test_original_objective_and_gradient():
    d=data()
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        value,gradient=objective_gradient(f['output_case'].ravel(),d);assert_allclose(value,f['output_loss'],rtol=1e-13);assert_allclose(gradient,f['output_gradient'],rtol=1e-12,atol=1e-14)
def test_original_full_pipeline_and_physical_fit():
    d=data();m,p,s=reconstruct(d)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for a,b in zip(m,f['output_reconstruction']):assert np.sqrt(np.mean((a-b)**2))/np.ptp(b)<5e-6
    assert_allclose(m[2,[0,-1]],0,atol=1e-14);assert s['projected_gradient_max']<1e-7
    zero=forward(np.zeros_like(m),d);mask=d['validity']>0;assert np.sqrt(np.mean((p-d['radial_velocity'])[mask]**2))<.2*np.sqrt(np.mean((zero-d['radial_velocity'])[mask]**2))
def test_exact_difference_adjoint():
    a=np.sin(np.arange(4*5*6)).reshape(4,5,6);b=np.cos(np.arange(a.size)).reshape(a.shape)
    for axis in range(3):assert_allclose(np.sum(derivative(a,2.,axis)*b),np.sum(a*derivative_adjoint(b,2.,axis)),atol=1e-13)
def test_gradient_independent_direction():
    d=data();shape=(3,)+d['radial_velocity'].shape[1:];x=np.sin(np.arange(np.prod(shape))).reshape(shape);v=np.cos(np.arange(x.size)).reshape(shape);x[2,[0,-1]]=0;v[2,[0,-1]]=0;eps=1e-3;value,g=objective_gradient(x.ravel(),d);fd=(objective_gradient((x+eps*v).ravel(),d)[0]-objective_gradient((x-eps*v).ravel(),d)[0])/(2*eps);assert_allclose(fd,g@v.ravel(),rtol=1e-8,atol=1e-8)
def test_uniform_anelastic_case():
    d=data();m=np.zeros((3,)+d['radial_velocity'].shape[1:]);m[0]=3;m[1]=-2;assert_allclose(divergence(m,d),0,atol=1e-14)
def test_readme_contract():
    d=data();text=(ROOT/'README.md').read_text(encoding='utf8')
    for k in d:assert '| '+k+' |' in text
    assert '5000' in text and '10000' in text and 'zero-mean' in text
