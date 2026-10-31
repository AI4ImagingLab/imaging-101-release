from pathlib import Path
import numpy as np
from src.preprocessing import load_data,find_points,extract_spectrum,interpolate_time
from src.physics_model import arc_coordinates,VIPA,setup
from src.solvers import reconstruct,calibration_regions
ROOT=Path(__file__).resolve().parents[2]

def test_independent_geometry_and_extraction_stages():
    raw=load_data(ROOT/'data/raw_data.npz');up=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    for i,k in enumerate(['1','2']):
        points=find_points(raw['calibration_images'][0,i]);np.testing.assert_array_equal(points,up['points_'+k])
        arc,circle=arc_coordinates(points);np.testing.assert_allclose(arc,up['arc_'+k],rtol=0,atol=1e-7)
        spectra=np.array([extract_spectrum(im,arc) for im in raw['calibration_images'][0,i]])
        np.testing.assert_allclose(spectra,up['calibration_spectra_'+k],rtol=1e-7,atol=1e-7)
        np.testing.assert_array_equal(calibration_regions(spectra),up['regions_'+k])

def test_full_pipeline_and_all_frozen_intermediates():
    raw=load_data(ROOT/'data/raw_data.npz');out,d=reconstruct(raw);up=np.load(ROOT/'test/fixtures/upstream_stages.npz');ref=np.load(ROOT/'evaluation/baseline_reference.npz')
    for k,v in ref.items():assert np.linalg.norm(out[k]-v)/np.linalg.norm(v)<=1e-5,k
    for i,k in enumerate(['1','2']):
        np.testing.assert_allclose(d['calibration_peak_positions'][i],up['calibration_peak_positions_'+k],rtol=1e-5)
        np.testing.assert_allclose(out['calibration_frequency'][0,i],up['calibration_frequency_'+k],rtol=0,atol=1e4)
    for i in range(10):
        np.testing.assert_allclose(out['extracted_spectra'][0,i],up['spectra_'+str(i)],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(out['frequency_axis'][0,i],up['frequency_'+str(i)],rtol=0,atol=1e4)
    # Every source result is consumed, including diagnostic time, mean camera intensity and drift.
    pars=out['peak_parameters'][0]
    for offset,kind in [(0,'brillouin'),(2,'rayleigh')]:
        for j,q in enumerate(['position_f','fwhm_f','intensity','offset']):
            np.testing.assert_allclose(pars[:,:,offset:offset+2,j],up['result_'+kind+'_peak_'+q].reshape(10,2,2),rtol=1e-5)
    np.testing.assert_allclose(out['reconstruction'],up['result_brillouin_shift_f'].reshape(1,10,2,2),rtol=1e-5)
    np.testing.assert_allclose(pars[:,:,2:,0]-pars[0:1,:,2:,0],up['result_rayleigh_shift'].reshape(10,2,2),rtol=1e-5,atol=1e4)
    np.testing.assert_allclose(raw['measurement_time'][0,:,None]+raw['exposure'][0,:,None]*np.arange(2),up['result_time'].reshape(10,2))
    np.testing.assert_allclose(raw['images'][0].mean(axis=(-1,-2)),up['result_intensity'].reshape(10,2))
