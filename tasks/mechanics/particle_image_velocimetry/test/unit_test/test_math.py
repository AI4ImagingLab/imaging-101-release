import numpy as np
import pytest
from src.preprocessing import windows,normalize,load_frames
from src.physics_model import correlate,field_coordinates
from src.solvers import solve,subpixel_peak,peak_confidence

def test_window_geometry():
    a=np.arange(369*511).reshape(369,511)
    w=windows(a);x,y=field_coordinates()
    assert w.shape==(70,64,64)
    np.testing.assert_array_equal(w[-1],a[288:352,432:496])
    assert x[0,0]==32 and x[0,-1]==464 and y[-1,0]==320

def test_normalize_constants_and_input_preservation():
    a=np.arange(24.).reshape(2,3,4);copy=a.copy();b=normalize(a)
    np.testing.assert_array_equal(a,copy)
    np.testing.assert_allclose(b.mean((-2,-1)),0,atol=1e-15)
    np.testing.assert_allclose(b.std((-2,-1)),1)
    assert not normalize(np.ones((2,3,4))).any()

def test_gaussian_peak_known_fractional_center():
    y,x=np.mgrid[:17,:17];c=np.exp(-((y-7.3)**2+(x-9.2)**2)/5)
    np.testing.assert_allclose(subpixel_peak(c),[7.3,9.2],atol=1e-7)

def test_negative_neighbors_parabolic_and_boundary():
    c=np.full((7,7),-1.);c[3,3]=2;c[4,3]=1;c[3,4]=1
    np.testing.assert_allclose(subpixel_peak(c),[3.25,3.25])
    c=np.zeros((7,7));c[0,0]=1
    assert np.isnan(subpixel_peak(c)).all() and peak_confidence(c)==0

def test_confidence_known_peaks():
    c=np.zeros((15,15));c[7,7]=8;c[3,3]=2
    assert peak_confidence(c)==4
    c[0,0]=6
    assert peak_confidence(c)==0

def test_bad_dt():
    with pytest.raises(ValueError):solve(np.zeros((64,64)),np.zeros((64,64)),dt=0)

def test_readme_constants():
    from pathlib import Path
    readme=Path('README.md').read_text(encoding='utf-8')
    for text in ['369, 511','32 + 48 j','32 + 48 i','64 by 64','32 by 32','0.0000001']:
        assert text in readme
    a,b=load_frames('data/raw_data.npz');assert a.shape==b.shape==(369,511)
