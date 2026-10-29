from pathlib import Path
import json
import numpy as np
from src.preprocessing import load_observations,calibration_levels,valid_pixels
from src.physics_model import normalize_ratio,invert_response
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]


def test_original_imagej_independent_calibration_means():
    _,low,high=load_observations(ROOT/'data/raw_data.npz')
    lower,upper,means5,means7=calibration_levels(low,high)
    reference=json.loads((ROOT/'test/fixtures/original_calibration_statistics.json').read_text())
    np.testing.assert_array_equal(means5,reference['image_means']['calibration5'])
    np.testing.assert_array_equal(means7,reference['image_means']['calibration7'])
    assert lower==reference['lower_ratio'] and upper==reference['upper_ratio']


def test_original_jython_normalization_inverse_and_clipping_stages():
    f=json.loads((ROOT/'test/fixtures/original_scalar_stages.json').read_text())
    normalized=normalize_ratio(f['input_ratio'],f['lower_ratio'],f['upper_ratio'])
    np.testing.assert_array_equal(normalized,f['output_normalized'])
    ph,clipped=invert_response(normalized)
    np.testing.assert_allclose(ph,f['output_pH_float64'],rtol=0,atol=2e-14)
    np.testing.assert_array_equal(ph.astype(np.float32),np.asarray(f['output_pH_float32'],dtype=np.float32))
    assert clipped[0] and clipped[-1] and not clipped[3]


def test_all_43_native_original_maps():
    ratios,low,high=load_observations(ROOT/'data/raw_data.npz');output,lower,upper,rates=reconstruct(ratios,low,high)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:reference=f['reconstruction'][0]
    mask=valid_pixels(ratios)
    assert np.array_equal(output[~mask],reference[~mask]) and np.isfinite(output).all()
    np.testing.assert_allclose(output[mask],reference[mask],rtol=0,atol=1e-6)
    for a,b,m in zip(output,reference,mask):
        assert np.sqrt(np.mean((a[m].astype(float)-b[m])**2))/np.ptp(b[m])<=1e-6
