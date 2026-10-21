import numpy as np
from src.physics_model import forward,adjoint,dirty_spectrum
from src.solvers import hogbom_clean

def test_complex_adjoint_identity():
    rng=np.random.default_rng(42)
    l2=np.linspace(.07,.14,41);phi=np.arange(-300.,301.,3.)
    x=rng.normal(size=len(phi))+1j*rng.normal(size=len(phi))
    y=rng.normal(size=len(l2))+1j*rng.normal(size=len(l2))
    np.testing.assert_allclose(np.vdot(forward(x,l2,phi,.1),y),np.vdot(x,adjoint(y,l2,phi,.1)),rtol=1e-12)

def test_thin_screen_phase_sign_scale_and_reference_wavelength():
    l2=np.linspace(.07,.14,41);phi=np.arange(-300.,301.,3.);lambda0=float(l2.mean())
    x=np.zeros(len(phi),complex);x[np.where(phi==120)[0][0]]=2*np.exp(.8j)
    p=forward(x,l2,phi,lambda0)
    np.testing.assert_allclose(p,2*np.exp(.8j+240j*(l2-lambda0)),rtol=1e-12)
    d=dirty_spectrum(p,l2,phi,np.ones(len(l2)),lambda0)
    assert phi[np.argmax(abs(d))]==120
    np.testing.assert_allclose(d[phi==120],2*np.exp(.8j),rtol=1e-12)

def test_clean_does_not_create_signal_below_cutoff():
    phi=np.arange(-9.,12.,3.);phi2=np.arange(-21.,24.,3.)
    dirty=np.full(len(phi),.001+.002j);response=np.exp(-phi2**2/10)
    clean,cc,residual,n=hogbom_clean(dirty,phi,response,phi2,3.,.01)
    np.testing.assert_array_equal(clean,dirty);np.testing.assert_array_equal(cc,0)
    assert n==0

def test_clean_residual_is_observation_operator_residual():
    from src.preprocessing import load_observations
    from src.solvers import reconstruct,PHI
    output,data=reconstruct(**load_observations())
    predicted=forward(output['components'][0],data['lambda2'],PHI,data['lambda0'])
    residual=dirty_spectrum(data['polarization']-predicted,data['lambda2'],PHI,data['weights'],data['lambda0'])
    np.testing.assert_allclose(residual,data['residual'],rtol=1e-10,atol=1e-13)

def test_visualization_complex_metric_matches_known_value():
    from src.visualization import relative_l2
    assert relative_l2(np.array([2+2j]),np.array([1+1j]))==1.

def test_public_metadata_matches_exact_acquisition_and_output_grid():
    from pathlib import Path
    from src.preprocessing import load_observations,C
    from src.solvers import PHI
    d=load_observations();readme=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert C==299792458. and '299792458' in readme
    np.testing.assert_array_equal(d['frequency'][[0,-1]],[800000000.,1088000000.])
    np.testing.assert_array_equal(d['noise'],np.full(288,.1))
    assert PHI.shape==(401,) and PHI[0]==-600 and PHI[-1]==600
    for key in d:assert '| '+key+' | (1, 288) |' in readme
