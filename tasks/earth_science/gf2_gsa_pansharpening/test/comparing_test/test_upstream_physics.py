import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data
from src.physics_model import interpolate23,lowpass,degrade,RATIO
from src.solvers import reconstruct,solve,solve_sfim,downsample_pan
def test_upstream_interpolation_and_quantized_pipeline():
    data=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream.npz')
    np.testing.assert_allclose(interpolate23(data['multispectral']),f['output_interpolated'],rtol=1e-10,atol=1e-12)
    np.testing.assert_array_equal(reconstruct(data)['reconstruction'],f['output_reconstruction'])
def test_lowpass_conservation_translation_and_constant():
    rng=np.random.default_rng(4);x=rng.random((32,32))
    assert np.isclose(lowpass(x).sum(),x.sum(),rtol=1e-14)
    np.testing.assert_allclose(lowpass(np.ones((32,32))),1,atol=1e-15)
    np.testing.assert_allclose(lowpass(np.roll(x,3,axis=1)),np.roll(lowpass(x),3,axis=1),atol=1e-15)
def test_interpolation_dc_and_phase_insertion():
    c=interpolate23(np.ones((8,8,1)),4)
    np.testing.assert_allclose(c,1,atol=2e-11)
    impulse=np.zeros((8,8,1));impulse[2,3]=1;up=interpolate23(impulse,4)[...,0]
    assert np.unravel_index(np.argmax(up),up.shape)==(10,14)
def test_fusion_bounds_and_zero_detail_ablation():
    d=load_data(ROOT/'data/raw_data.npz');o=solve(d)
    assert o.shape==(256,256,4) and o.min()>=0 and o.max()<=1
    np.testing.assert_allclose(solve(d,detail_gain=0),np.clip(interpolate23(d['multispectral']),0,1),rtol=1e-12,atol=1e-12)
    assert degrade(o).shape==(64,64,4)
def test_readme_constants():
    assert '| Resolution ratio | 4 |' in (ROOT/'README.md').read_text(encoding='utf8')

def test_original_pan_resampling_and_sfim_failure_comparison():
    d=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream.npz')
    centered=d['panchromatic']-d['panchromatic'].mean()
    np.testing.assert_array_equal(downsample_pan(centered[...,0]),f['output_downsampled_pan'])
    failure=np.load(ROOT/'test/fixtures/sfim_upstream.npz')['output_reconstruction']
    np.testing.assert_array_equal(np.uint8(solve_sfim(d)*255)[None],failure)

def test_gsa_constant_intensity_is_rejected_before_division():
    import pytest
    d={'multispectral':np.ones((64,64,4),dtype='float32')*.4,'panchromatic':np.ones((256,256,1),dtype='float32')*.4}
    with pytest.raises(ValueError,match='zero-variance'):solve(d)

def test_frozen_upstream_regression_and_covariance_stages():
    from src.solvers import regression_intensity,detail_gains
    d=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream.npz')
    hs=d['multispectral'];up=interpolate23(hs);centered=up-up.mean((0,1));low=hs-hs.mean((0,1))
    intensity,alpha=regression_intensity(low,centered,f['output_downsampled_pan'][...,None])
    np.testing.assert_allclose(alpha,f['output_alpha'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(intensity,f['output_intensity'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(detail_gains(intensity,centered),f['output_gains'],rtol=1e-10,atol=1e-12)
    design=np.concatenate((low,np.ones((64,64,1))),-1).reshape(-1,5)
    residual=design@alpha-f['output_downsampled_pan'].reshape(-1,1)
    assert np.linalg.norm(design.T@residual)<1e-10

def test_covariance_sample_population_factor_independently():
    from src.solvers import detail_gains
    intensity=np.arange(16,dtype=float).reshape(4,4,1)
    bands=np.repeat(intensity,4,-1)*np.arange(1,5)
    np.testing.assert_allclose(detail_gains(intensity,bands),np.arange(1,5)*16/15,rtol=1e-14)
