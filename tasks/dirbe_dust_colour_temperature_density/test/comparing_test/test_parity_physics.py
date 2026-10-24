from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.physics_model import planck,flux,forward,jacobian,bandpass_prediction,C,K
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_planck_and_flux():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for i,(t,m) in enumerate(zip(f['temperatures'],f['masses'])):
            assert_allclose(planck(f['wavelengths'],t),f['planck'][i],rtol=2e-14)
            assert_allclose(flux(f['wavelengths'],t,m),f['flux'][i],rtol=2e-14)
def test_complete_original_pipeline():
    data=load_data(ROOT/'data/raw_data.npz');m,p=reconstruct(data)
    with np.load(ROOT/'test/fixtures/upstream_pipeline.npz',allow_pickle=False) as f:
        assert_allclose(data['intensity'],f['input_intensity'],rtol=0,atol=0)
        assert_allclose(m,f['output_reconstruction'],rtol=1e-8,atol=1e-10)
        assert_allclose(p,f['output_prediction'],rtol=1e-8,atol=1e-8)
def test_rayleigh_jeans_limit():
    w=np.array([1e8]);t=20.;nu=C/(w*1e-6);expected=2*K*t*nu**2/C**2
    assert_allclose(planck(w,t),expected,rtol=4e-6)
def test_analytic_jacobian():
    w=np.array([100.,140.,240.]);t=np.array([[18.]]);s=np.array([[.001]])
    j=jacobian(w,t,s)
    assert_allclose(j[...,0],(forward(w,t+1e-4,s)-forward(w,t-1e-4,s))/2e-4,rtol=2e-9)
    assert_allclose(j[...,1],(forward(w,t,s+1e-8)-forward(w,t,s-1e-8))/2e-8,rtol=2e-10)
def test_column_linearity_and_positive_planck():
    w=np.array([100.,140.,240.]);t=np.array([[18.]]);s=np.array([[.001]])
    assert_allclose(forward(w,t,2*s),2*forward(w,t,s),rtol=1e-14)
    assert np.all(planck(w,20)>planck(w,15))
def test_narrow_passband_limit():
    w=np.array([140.]);rw=np.linspace(139.999,140.001,101);r=np.ones((1,101));t=np.array([[18.]]);s=np.array([[.001]])
    assert_allclose(bandpass_prediction(w,t,s,rw,r),forward(w,t,s),rtol=1e-9)
def test_readme_constants_and_shapes():
    d=load_data(ROOT/'data/raw_data.npz');text=(ROOT/'README.md').read_text(encoding='utf8')
    for key in d:assert '| '+key+' |' in text
    for value in ['6.62606957e-34','1.3806488e-23','0.192','100,140,240']:assert value in text
    assert not any(x in text.lower() for x in ['github.com','github.io','githubusercontent'])
