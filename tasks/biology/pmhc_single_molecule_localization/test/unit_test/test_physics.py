from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import bandpass_filter,load_observations
from src.solvers import locate,find_peaks_yx
from src.physics_model import gaussian_image
ROOT=Path(__file__).resolve().parents[2]

def test_dc_rejection_and_zero_boundary():
    result=bandpass_filter(np.full((32,32),200.))
    assert np.max(np.abs(result))<1e-12
    assert np.count_nonzero(result[:3])==0

def test_integer_centroid_flux_and_shift():
    image=gaussian_image((64,64),np.array([[25.,30.]]),[20000.],[1.])
    assert abs(image.sum()-20000)<1e-3
    table=locate(image,4,10,1000,bandpass=False)
    assert len(table)==1
    np.testing.assert_allclose(table[0,:2],[25.,30.],atol=1e-12)
    translated=np.roll(image,(2,3),axis=(0,1));t2=locate(translated,4,10,1000,bandpass=False)
    np.testing.assert_allclose(t2[0,:2]-table[0,:2],[3.,2.],atol=1e-12)
    np.testing.assert_allclose(t2[0,2:],table[0,2:],rtol=1e-12,atol=1e-12)
    assert table[0,4]<1e-6

def test_subpixel_centroid_and_radius():
    image=gaussian_image((64,64),np.array([[25.3,30.6]]),[20000.],[1.])
    t=locate(image,4,10,1000,bandpass=False)
    np.testing.assert_allclose(t[0,:2],[25.3,30.6],atol=.01)
    assert 1.35<t[0,3]<1.65

def test_peak_boundaries_and_public_data():
    frame=np.zeros((16,16));frame[1,1]=1000;frame[8,8]=1000
    np.testing.assert_array_equal(find_peaks_yx(frame,3,100),[[8,8]])
    raw=load_observations(ROOT/'data/raw_data.npz');assert raw.shape==(1,10,128,128)
