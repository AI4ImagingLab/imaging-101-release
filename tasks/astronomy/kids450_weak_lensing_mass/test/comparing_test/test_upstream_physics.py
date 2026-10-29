import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data
from src.solvers import grid_catalogue,reconstruct,RA_CENTRE,DEC_CENTRE,GRID_SIZE
from src.physics_model import forward,inverse
def test_frozen_upstream_stage_and_full_pipeline():
    data=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream.npz')
    g1,g2,_,_=grid_catalogue(data)
    np.testing.assert_allclose(g1,f['output_g1'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(g2,f['output_g2'],rtol=1e-10,atol=1e-12)
    e,b=inverse(g1,g2)
    np.testing.assert_allclose(e,f['output_unsmoothed_e'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(b,f['output_unsmoothed_b'],rtol=1e-10,atol=1e-12)
    for k,v in reconstruct(data).items():np.testing.assert_allclose(v,f['output_'+k],rtol=1e-10,atol=1e-12)
def test_analytic_potential_spin2_and_mass_sheet():
    y,x=np.mgrid[:64,:64];q=2*np.pi*(3*x+2*y)/64;e=np.cos(q);b=np.zeros_like(e)
    g1,g2=forward(e,b)
    np.testing.assert_allclose(g1,5/13*e,atol=2e-14)
    np.testing.assert_allclose(g2,12/13*e,atol=2e-14)
    er,br=inverse(g1,g2);np.testing.assert_allclose(er,e,atol=3e-14);np.testing.assert_allclose(br,0,atol=3e-14)
    z1,z2=forward(np.ones((64,64)),b);assert np.max(abs(z1))+np.max(abs(z2))==0
def test_adjoint_inner_product():
    rng=np.random.default_rng(4);e,b,g1,g2=rng.normal(size=(4,64,64))
    f1,f2=forward(e,b);a1,a2=inverse(g1,g2)
    assert np.allclose(np.sum(f1*g1+f2*g2),np.sum(e*a1+b*a2),rtol=1e-12,atol=1e-10)
def test_calibration_ensemble_is_not_individual_division():
    d={'positions':np.array([[129.,-.5],[129.,-.5]]),'ellipticity':np.array([[.1,.2],[.3,.4]]),'weight':np.array([1.,3.]),'multiplicative_bias':np.array([.1,-.2])}
    g1,g2,n,w=grid_catalogue(d);pixel=np.unravel_index(np.argmax(n),n.shape)
    assert np.isclose(g1[pixel],1./3.5);assert np.isclose(g2[pixel],1.4/3.5)
def test_readme_constants():
    s=(ROOT/'README.md').read_text(encoding='utf8');assert str(RA_CENTRE) in s and str(DEC_CENTRE) in s and str(GRID_SIZE) in s

def test_mixed_nyquist_mode_has_declared_real_projection_loss():
    y,x=np.mgrid[:64,:64];mode=(-1.)**x*np.cos(2*np.pi*3*y/64)
    g1,g2=forward(mode,np.zeros_like(mode));recovered,_=inverse(g1,g2)
    p1=(32**2-3**2)/(32**2+3**2)
    np.testing.assert_allclose(recovered,p1*p1*mode,atol=2e-14)
    assert np.linalg.norm(g2)<1e-13
    assert np.isclose(np.linalg.norm(recovered-mode)/np.linalg.norm(mode),1-p1*p1,rtol=1e-12)
