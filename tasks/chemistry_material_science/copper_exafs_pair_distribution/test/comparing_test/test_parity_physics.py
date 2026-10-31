from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from src.preprocessing import load_observations
from src.physics_model import *
F=Path(__file__).resolve().parents[1]/'fixtures/upstream.npz'

def test_raw_and_scattering_tables_exact():
    d=load_observations()
    with np.load(F,allow_pickle=False) as f:
        for key,suffix in [('k','k'),('chi','chi'),('scattering_k','kernel_k'),('scattering_phase','kernel_phase'),('scattering_amplitude','kernel_amplitude'),('real_momentum','kernel_re_p'),('mean_free_path','kernel_mfp')]:np.testing.assert_array_equal(d[key],f['input_'+suffix])
        assert REFF==float(f['config_reff']) and DEGENERACY==float(f['config_degeneracy'])

def test_upstream_forward_window_transform_residual():
    d=load_observations();k=model_grid(d);spl=scattering_splines(d)
    with np.load(F,allow_pickle=False) as f:
        np.testing.assert_array_equal(k,f['input_model_k'])
        np.testing.assert_allclose(forward(INITIAL,k,spl),f['output_initial'],rtol=2e-12,atol=1e-14)
        np.testing.assert_allclose(forward(f['output_parameters'],k,spl),f['output_model'],rtol=2e-12,atol=1e-14)
        np.testing.assert_allclose(kaiser_window(),f['output_window'],rtol=1e-13,atol=2e-14)
        chi=np.interp(k,d['k'],d['chi'])
        np.testing.assert_allclose(transform(chi),f['output_data_transform'],rtol=2e-12,atol=1e-13)
        np.testing.assert_allclose(residuals(f['output_parameters'],k,chi,spl),f['output_residual'],rtol=2e-10,atol=1e-13)

def test_independent_fft_sign_and_scale():
    j=90;x=np.zeros(500);x[j]=1
    expected=KSTEP/np.sqrt(np.pi)*(j*KSTEP)**2*kaiser_window()[j]*np.exp(-2j*np.pi*j*np.arange(1024)/NFFT)
    np.testing.assert_allclose(transform(x),expected,rtol=2e-12,atol=1e-12)

def test_amplitude_linearity_and_pair_moments():
    d=load_observations();k=model_grid(d);s=scattering_splines(d)
    p=np.array([.9,.4,.0034,.002]);p2=p.copy();p2[0]*=2
    np.testing.assert_allclose(forward(p2,k,s),2*forward(p,k,s),rtol=2e-13,atol=1e-13)
    r=np.linspace(.1,5,20001);v=pair_density(p,r)
    n=trapezoid(v,r);mean=trapezoid(r*v,r)/n;variance=trapezoid((r-mean)**2*v,r)/n
    np.testing.assert_allclose([n,mean,variance],[12,REFF+p[3],p[2]],rtol=1e-12,atol=1e-13)

def test_fresh_full_inversion():
    from src.solvers import reconstruct
    x,info=reconstruct(load_observations())
    with np.load(F,allow_pickle=False) as f:
        assert np.max(np.abs(x['parameters'][0]-f['output_parameters'])/PARAMETER_SCALES)<2e-4
    assert info['success'] and info['rfactor']<.00125
