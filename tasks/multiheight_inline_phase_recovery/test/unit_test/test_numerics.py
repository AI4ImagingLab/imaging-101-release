import numpy as np
import importlib.util
from pathlib import Path
from src.physics_model import propagate,transfer,amplitude_residual
from src.preprocessing import amplitude_stack,DISTANCES
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_propagation_energy_inverse_and_adjoint():
    rng=np.random.default_rng(731);x=rng.normal(size=(32,32))+1j*rng.normal(size=(32,32));y=rng.normal(size=(32,32))+1j*rng.normal(size=(32,32))
    fx=propagate(x,50);back=propagate(y,-50)
    np.testing.assert_allclose(np.vdot(fx,y),np.vdot(x,back),rtol=5e-7)
    np.testing.assert_allclose(np.linalg.norm(fx),np.linalg.norm(x),rtol=5e-7)
    np.testing.assert_allclose(propagate(fx,-50),x,rtol=5e-7,atol=5e-7)
def test_constant_field_phase():
    x=np.ones((32,32),complex);h=transfer(x.shape,50)
    np.testing.assert_allclose(propagate(x,50),x*h[0,0],atol=1e-12)
def test_preprocessing_known_values():
    a=np.array([[[0,65535],[16384,32768]]],dtype='uint16')
    out=amplitude_stack(a);assert out.dtype==np.complex64
    np.testing.assert_allclose(abs(out)**2,a.astype('float32')/65535,atol=1e-7)
def test_multiplane_solver_reduces_mismatch_on_analytic_wave():
    y,x=np.mgrid[:32,:32];u=np.exp(0.2j*np.cos(2*np.pi*x/32))*np.exp(-((x-16)**2+(y-16)**2)/180)
    z=np.array([300.,350.,400.]);a=np.stack([abs(propagate(u,d)) for d in z]).astype('complex64')
    initial,_=reconstruct(a,0,z);fresh,h=reconstruct(a,20,z)
    assert amplitude_residual(fresh,a,z)<amplitude_residual(initial,a,z)
def load_judge():
    spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_metric_known_complex_cases():
    judge=load_judge();ref=np.array([1+0j,2+0j,3+0j])
    m=judge.compare(ref*1j,ref);assert m['ncc']>1-1e-14;assert m['nrmse']<1e-14
    m=judge.compare(2*ref,ref);np.testing.assert_allclose(m['nrmse'],np.sqrt(14/3)/2)
    assert judge.compare(np.zeros(3,complex),ref)['ncc']==0

def test_judge_and_visual_residual_agree():
    from src.physics_model import amplitude_residual
    a=np.load(ROOT/'test/fixtures/upstream_stages.npz')['output_preprocessing']
    u=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'][0]
    m=load_judge().evaluate(ROOT/'evaluation/baseline_reference.npz')
    np.testing.assert_allclose(m['metrics']['amplitude_residual'],amplitude_residual(u,a,DISTANCES),rtol=1e-8,atol=1e-10)

def test_readme_acquisition_constants():
    from src.preprocessing import PIXEL_SIZE,WAVELENGTH,DISTANCES
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert f'| Wavelength | {WAVELENGTH} |' in text
    assert f'| Pixel spacing | {PIXEL_SIZE} |' in text
    assert ', '.join(str(int(z)) for z in DISTANCES) in text
    forbidden=('github.com','github.io','githubusercontent.com','codeload.github.com','git.io')
    assert not any(x in text.lower() for x in forbidden)

def test_unsupported_grid_shapes():
    import pytest
    from src.physics_model import wave_grid
    for shape in [(31,32),(32,31),(32,), (0,32)]:
        with pytest.raises(ValueError,match='even two-dimensional'):wave_grid(shape)

def test_prediction_preserves_numpy_distance_precision():
    from src.physics_model import predicted_amplitudes
    a=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'][0]
    distances=np.array([300.,650.],dtype='float64')
    expected=np.stack([abs(propagate(a,np.float64(d))) for d in distances])
    np.testing.assert_array_equal(predicted_amplitudes(a,distances),expected)
