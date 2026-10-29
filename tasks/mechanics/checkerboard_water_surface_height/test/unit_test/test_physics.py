import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.physics_model import frequencies,integrate_gradient,height_gradient,effective_height
from src.unwrapping import unwrap_reliability
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_analytic_fourier_surface():
    y,x=np.mgrid[:33,:35];h=2e-4*np.cos(2*np.pi*3*x/35)+1e-4*np.sin(2*np.pi*2*y/33);gradient=height_gradient(h,.0003);v=integrate_gradient(*gradient,.0003)
    np.testing.assert_allclose(v,h,rtol=1e-12,atol=1e-16)
def test_gradient_projection_normal_equations():
    rng=np.random.default_rng(7);g=rng.normal(size=(2,33,35));h=integrate_gradient(*g,.2);e=height_gradient(h,.2)-g;ky,kx=frequencies(h.shape,.2)
    divergence=1j*kx*np.fft.fft2(e[0])+1j*ky*np.fft.fft2(e[1]);np.testing.assert_allclose(divergence,0,atol=2e-11)
def test_even_grid_is_rejected():
    with pytest.raises(ValueError,match='odd'):integrate_gradient(np.zeros((32,32)),np.zeros((32,32)),1.)
def test_analytic_wrapped_surface():
    y,x=np.mgrid[:33,:35];truth=.2*x+.1*y+4*np.sin(2*np.pi*x/35)+2*np.cos(2*np.pi*y/33);wrapped=np.angle(np.exp(1j*truth));v=unwrap_reliability(wrapped);difference=v-truth;difference-=np.median(difference)
    np.testing.assert_allclose(difference,0,atol=1e-12)
def test_known_metrics_and_geometry():
    r=np.array([0.,1.,2.]);m=judge.compare(r*2,r);np.testing.assert_allclose(m['nrmse'],np.sqrt(5/3)/2);assert abs(m['ncc']-1)<1e-14
    assert effective_height()==.0329956084466042

def test_public_contract_and_constants():
    from src.preprocessing import load_data,SQUARE_SIZE
    r,d=load_data(ROOT/'data/raw_data.npz');assert r.shape==(641,641) and d.dtype==np.float32 and SQUARE_SIZE==.0022
    text=(ROOT/'README.md').read_text(encoding='utf8');assert '0.0329956084466042' in text and '0.00031747269890795626' in text
    assert not any(x in text.lower() for x in ['github.com','github.io','githubusercontent.com','codeload.github.com','git.io'])

def test_original_even_grid_forensic_case_and_valid_odd_case():
    z=np.load(ROOT/'test/fixtures/upstream_integrator_forensics.npz')
    error=np.linalg.norm(z['output_upstream_16']-z['output_analytic_16'])/np.linalg.norm(z['output_analytic_16']);np.testing.assert_allclose(error,.5,rtol=1e-12)
    local=integrate_gradient(z['input_gx_17'],z['input_gy_17'],1.)
    np.testing.assert_allclose(local,z['output_upstream_17'],atol=1e-14);np.testing.assert_allclose(local,z['output_analytic_17'],atol=1e-13)
def test_visualization_and_judge_phase_metrics_agree():
    from src.preprocessing import load_data
    from src.physics_model import carriers,phases
    from src.visualization import phase_residual,metrics
    reference,image=load_data(ROOT/'data/raw_data.npz');c=carriers(reference);diagnostics={'carriers':c,'phases':phases(image,c,False)}
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:h=z['reconstruction'][0]
    np.testing.assert_allclose(phase_residual(h,diagnostics)[0],judge.observation_residual(h),rtol=1e-12)
    r=np.array([0.,1.,2.]);np.testing.assert_allclose(list(metrics(2*r,r).values()),list(judge.compare(2*r,r).values()),rtol=1e-14)

def test_visual_zero_and_degenerate_metrics():
    from src.visualization import metrics
    r=np.array([0.,1.,2.]);assert metrics(np.zeros(3),r)['ncc']==0
    with pytest.raises(ValueError,match='degenerate'):metrics(r,np.ones(3))
