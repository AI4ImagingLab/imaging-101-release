import numpy as np
import pytest
from src.physics_model import optical_constants,forward,jacobian
from src.visualization import metrics
from src.solvers import fit_pixel
def test_zero_oxide_matches_independent_direct_fresnel():
    theta=np.linspace(0,.7,13);n0=1.33;n1=3.866+.017933j
    transmitted=np.arcsin(n0*np.sin(theta)/n1);r=(n0*np.cos(theta)-n1*np.cos(transmitted))/(n0*np.cos(theta)+n1*np.cos(transmitted))
    c=optical_constants(theta,oxide_nm=0)
    np.testing.assert_allclose(c[:,0]+1j*c[:,1],r,rtol=1e-13,atol=1e-13)
def test_passive_reflectance_and_optical_phase_units():
    theta=np.linspace(0,.8,20);c=optical_constants(theta)
    assert np.all(c[:,0]**2+c[:,1]**2<=1)
    np.testing.assert_allclose(c[:,2],4*np.pi*1.33/642*np.cos(theta))
@pytest.mark.parametrize('column',[0,1,2])
def test_analytic_derivative(column):
    c=optical_constants(np.linspace(.1,.7,32));p=np.array([850.,3100.,87.]);d=np.zeros(3);d[column]=1e-4
    fd=(forward(c,p+d)-forward(c,p-d))/(2e-4)
    np.testing.assert_allclose(jacobian(c,p)[:,column],fd,atol=1e-7,rtol=1e-6)
def test_noiseless_identifiable_inverse():
    c=optical_constants(np.linspace(.1,.7,32));truth=np.array([850.,3100.,87.]);p,it,ok,sse=fit_pixel(c,forward(c,truth))
    assert ok and sse<1e-12;np.testing.assert_allclose(p,truth,atol=1e-5,rtol=1e-7)
def test_metrics_known_noncentered_case():
    r=np.array([1.,2.,3.]);e=2*r;m=metrics(e,r)
    assert m['ncc']==pytest.approx(1);assert m['nrmse']==pytest.approx(np.sqrt(14/3)/2)
def test_degenerate_metric_reference_rejected():
    with pytest.raises(ValueError):metrics(np.ones(3),np.ones(3))
def test_independent_judge_metric_known_case():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('standalone_judge',Path(__file__).resolve().parents[2]/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    r=np.array([1.,2.,3.]);result=m.score(2*r,r)
    assert result['ncc']==pytest.approx(1);assert result['nrmse']==pytest.approx(np.sqrt(14/3)/2)
def test_readme_calibration_matches_code_and_bound_inputs():
    from pathlib import Path
    from src.physics_model import N_BUFFER,OXIDE_NM,WAVELENGTH_NM,N_OXIDE,N_SILICON
    root=Path(__file__).resolve().parents[2];readme=(root/'README.md').read_text(encoding='utf8')
    assert (N_BUFFER,OXIDE_NM,WAVELENGTH_NM,N_OXIDE,N_SILICON)==(1.33,1924.9,642.,1.4719,3.866+.017933j)
    for value in ['1924.9','642.0','1.4719','3.8660+0.017933i']:assert value in readme
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as f:np.testing.assert_array_equal(f['angles'][0],np.linspace(5,43.75,32)*2*np.pi/360)
