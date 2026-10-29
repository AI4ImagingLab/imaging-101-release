"""Frozen original stages, captured before local extraction."""
from pathlib import Path
import numpy as np
from src.preprocessing import load_data,calibration_fit,polarization_vectors,subset_indices
from src.physics_model import angular_coupling,transfer_fields,forward_linear
ROOT=Path(__file__).resolve().parents[2]
F=np.load(ROOT/'test/fixtures/upstream_stages.npz')

def test_selected_angular_gaunt_kernel():
    np.testing.assert_allclose(angular_coupling(),F['angular_coupling'],rtol=1e-13,atol=2e-15)

def test_native_calibration_and_polarization():
    raw=load_data(ROOT/'data/raw_data.npz');offset,means,_,_=calibration_fit(raw['lake_calibration'][0])
    assert offset==float(F['calibration_offset'])
    np.testing.assert_array_equal(means,F['calibration_averages'])
    ia,ib=subset_indices('ALL')
    np.testing.assert_allclose(polarization_vectors(ia,ib,offset),F['pols'],rtol=0,atol=1e-15)

def test_native_transfer_functions():
    h=transfer_fields(F['pols'],(77,55,80))
    for k in ['Hxy','Hyz','Hx','Hz']:
        np.testing.assert_allclose(h[k],F[k],rtol=3e-7,atol=5e-8)

def test_native_uniform_calibration_response():
    h=transfer_fields(F['pols'],(77,55,80));f=np.zeros((77,55,80,15));f[...,0]=1
    response=forward_linear(f,h);response/=np.max(response)
    np.testing.assert_allclose(response.mean(axis=(0,1,2)),F['model_averages'],rtol=3e-7,atol=1e-8)
