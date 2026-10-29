from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.preprocessing import linear_fit,native_to_vertical,PIXEL_MM,FREQUENCIES
from src.physics_model import thermal_profile,diffusivity_from_slopes,fit_forward
from src.visualization import relative_error,profile_figure
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_ols_and_forward():
    x=np.linspace(0,1,100);c,e=linear_fit(x,3*x-2);np.testing.assert_allclose(c,[3,-2],atol=1e-14);assert e<1e-14
    np.testing.assert_allclose(fit_forward(x,c),3*x-2,atol=1e-14)
def test_harmonic_pde_and_loss_cancellation():
    x=np.linspace(.1,.7,5001);a=.83;f=6.3;loss=14.;u=thermal_profile(x,a,f,loss)
    dx=x[1]-x[0];second=(u[2:]-2*u[1:-1]+u[:-2])/dx**2
    np.testing.assert_allclose(a*second,(loss+2j*np.pi*f)*u[1:-1],rtol=1e-6)
    ma=linear_fit(x,np.log(abs(u)))[0][0];mp=linear_fit(x,np.unwrap(np.angle(u)))[0][0]
    np.testing.assert_allclose(diffusivity_from_slopes(ma,mp,f),a,rtol=1e-12)
def test_units_geometry():
    a,p=native_to_vertical(np.array([[0.,1.],[2.,3.]]),np.ones((2,2))*180)
    np.testing.assert_array_equal(a,[[1,3],[.001,2]]);np.testing.assert_allclose(p,np.pi)
    assert PIXEL_MM==.0013;np.testing.assert_array_equal(FREQUENCIES,[7.547,6.154])
def test_metric_known_case():
    assert relative_error(np.array([2.,6.]),np.array([1.,3.]))==1
    with pytest.raises(ValueError):relative_error([1],[0])
def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k] for k in judge.SHAPES}
@pytest.mark.parametrize('bad',['missing','shape','integer','float16','complex','nan','inf'])
def test_bad_submission(tmp_path,bad):
    d=good()
    if bad=='missing':del d['reconstruction']
    elif bad=='shape':d['reconstruction']=d['reconstruction'][0]
    elif bad=='integer':d['reconstruction']=d['reconstruction'].astype(int)
    elif bad=='float16':d['reconstruction']=d['reconstruction'].astype(np.float16)
    elif bad=='complex':d['reconstruction']=d['reconstruction'].astype(complex)
    else:d['reconstruction'][0,0,0]=float(bad)
    p=tmp_path/'bad.npz';np.savez(p,**d);r=judge.evaluate(p);assert r['status']=='error' and r['metrics']=={} and r['errors']
def test_missing_assets_and_poor(tmp_path):
    assert judge.evaluate(tmp_path/'none')['status']=='error'
    p=tmp_path/'p.npz';np.savez(p,**good());assert judge.evaluate(p,tmp_path)['status']=='error'
    d=good();d['reconstruction']*=2;np.savez(p,**d);assert judge.evaluate(p)['status']=='fail'
def test_judge_known_reference(tmp_path):
    p=tmp_path/'p.npz';d=good();np.savez(p,**d);r=judge.evaluate(p);assert r['passed']
    assert r['metrics']['relative_error']==0 and r['metrics']['coefficient_relative_error']==0
    assert r['metrics']['log_amplitude_rmse']==pytest.approx(.07901680999052618)

def test_ground_truth_precedence(tmp_path):
    import shutil
    for name in ('baseline_reference.npz','metrics.json'):shutil.copyfile(ROOT/'evaluation'/name,tmp_path/name)
    d=good();p=tmp_path/'prediction.npz';np.savez(p,**d)
    d['reconstruction']*=2;np.savez(tmp_path/'ground_truth.npz',**d)
    r=judge.evaluate(p,tmp_path)
    assert r['status']=='fail' and r['metrics']['relative_error']==.5
    # Observation scoring still uses fixed original extractions of public measurements.
    assert r['metrics']['log_amplitude_rmse']==pytest.approx(.07901680999052618)
