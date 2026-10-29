from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data
from src.physics_model import synthesize_response,humlicek,degrade,LINES
from src.solvers import INITIAL,SCALE,lm_block,smooth_model

def fixture():return np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False)

def test_exact_observation_preprocessing():
    d=load_data()
    with fixture() as f:
        np.testing.assert_allclose(d['regions'][0],f['config_fine1'],atol=0,rtol=0)
        np.testing.assert_allclose(d['regions'][1],f['config_fine2'],atol=0,rtol=0)
        np.testing.assert_array_equal(d['indices'],f['output_indices'])

def test_original_spectra_and_nine_responses():
    d=load_data();s,r=synthesize_response(np.tile(INITIAL,(6,1)),d['regions'],d['psfs'])
    with fixture() as f:
        np.testing.assert_allclose(s,f['output_initial_spectra'].reshape(6,4,-1),rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(r,f['output_initial_response'].reshape(6,9,4,-1),rtol=1e-8,atol=1e-9)

def test_complete_pixel_updates_including_rejections():
    d=load_data();y=np.array([0,0,0,7,44,63]);x=np.array([0,15,83,80,69,95])
    m,s,q,_=lm_block(np.tile(INITIAL,(6,1)),d['observations'][y,x],d['sigma'],d['regions'],d['psfs'])
    with fixture() as f:
        np.testing.assert_allclose(m/SCALE,f['output_cycle1'][y,x]/SCALE,rtol=0,atol=1e-5)
        np.testing.assert_allclose(q,f['output_cycle1_chi2'][y,x],rtol=1e-8,atol=1e-8)

def test_full_image_spatial_smoothing():
    with fixture() as f:np.testing.assert_allclose(smooth_model(f['output_cycle1']),f['output_smooth'],atol=1e-10,rtol=1e-12)

def test_polarization_symmetries_and_continuum():
    d=load_data();m=INITIAL[None].copy();m[0,0]=0
    s,_=synthesize_response(m,d['regions'],d['psfs']);np.testing.assert_allclose(s[:,1:],0,atol=1e-13)
    m=INITIAL[None].copy();p,_=synthesize_response(m,d['regions'],d['psfs']);m[0,2]+=np.pi
    q,_=synthesize_response(m,d['regions'],d['psfs']);np.testing.assert_allclose(p,q,atol=1e-13)
    m[0,5]=0;s,_=synthesize_response(m,d['regions'],d['psfs']);np.testing.assert_allclose(s[:,0],m[0,7]+.93*m[0,8],atol=1e-13)

def test_response_independent_finite_difference():
    d=load_data();m=INITIAL[None].copy();s,r=synthesize_response(m,d['regions'],d['psfs'])
    # The source response uses exact Voigt identities with an approximate Voigt
    # value. Its finite-difference discrepancy is bounded by that approximation.
    for k in range(9):
        h=SCALE[k]*1e-5;p=m.copy();q=m.copy();p[:,k]+=h;q[:,k]-=h
        plus,_=synthesize_response(p,d['regions'],d['psfs'],with_response=False)
        minus,_=synthesize_response(q,d['regions'],d['psfs'],with_response=False)
        numerical=(plus-minus)/(2*h)
        np.testing.assert_allclose(numerical*SCALE[k],r[:,k]*SCALE[k],rtol=2e-3,atol=3e-5)

def test_zeeman_sum_rules_and_flux_conservation():
    for _,_,components in LINES:
        for pol in range(3):assert abs(sum(v for p,v,_ in components if p==pol)-1)<1e-14
    x=np.full((4,123),.7);np.testing.assert_allclose(degrade(x,np.ones(75)),x,atol=1e-15)
