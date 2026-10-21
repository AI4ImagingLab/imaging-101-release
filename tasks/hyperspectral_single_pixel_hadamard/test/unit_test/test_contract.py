import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations,spectral_bins
from src.physics_model import sequency_order,inverse
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_known_metric_cases():
    x=np.array([0.,1.,2.]);m=j.reference_metrics(x+1,x)
    np.testing.assert_allclose(m['nrmse'],.5)
    np.testing.assert_allclose(m['ncc'],8/(np.sqrt(14)*np.sqrt(5)))
    assert j.reference_metrics(np.zeros(3),x)['ncc']==0
    with pytest.raises(ValueError):j.reference_metrics(x,np.ones(3))
    from src.visualization import metric_values
    assert metric_values(x+1,x)==m

@pytest.mark.parametrize('case',['missing_key','shape','dtype','complex','nan','inf','overflow'])
def test_invalid_prediction_errors(case,tmp_path):
    if case=='missing_key':contents={'wrong':np.zeros(1)}
    else:
        if case=='shape':x=np.zeros((1,64,64,3))
        else:
            dtype=np.float32 if case=='dtype' else np.complex128 if case=='complex' else np.float64
            x=np.zeros(j.SHAPE,dtype=dtype)
            if case in ['nan','inf','overflow']:x.flat[0]={'nan':np.nan,'inf':np.inf,'overflow':1e308}[case]
        contents={'reconstruction':x}
    p=tmp_path/'bad.npz';np.savez_compressed(p,**contents)
    r,code=j.evaluate(p)
    assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_poor_result_and_full_judge(tmp_path):
    p=tmp_path/'poor.npz';np.savez_compressed(p,reconstruction=np.zeros(j.SHAPE))
    r,code=j.evaluate(p);assert code==1 and r['status']=='fail' and not r['passed']
    r,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert code==0 and r['passed']
    assert r['metrics']['relative_measurement_residual']==0

def test_missing_assets_and_prediction(tmp_path):
    assert j.evaluate(tmp_path/'missing.npz')[1]==2
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)[1]==2

def test_truth_precedence_and_invalid_truth(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:truth=f['reconstruction']
    np.savez_compressed(tmp_path/'ground_truth.npz',reconstruction=truth)
    np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=np.zeros(1))
    monkeypatch.setattr(j,'observation_residual',lambda x:0.)
    assert j.evaluate(tmp_path/'ground_truth.npz',tmp_path)[1]==0
    np.savez(tmp_path/'ground_truth.npz',reconstruction=np.zeros(1))
    r,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and r['metrics']=={}

def test_cli_foreign_workdir_and_report_failure(tmp_path):
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/r.json')]
    assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==0
    report=json.loads((tmp_path/'nested/r.json').read_text());assert report['status']=='pass'
    args[-1]=str(tmp_path)
    assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==2

def test_small_validation_and_spectral_binning():
    with pytest.raises(ValueError):sequency_order(7)
    with pytest.raises(ValueError):inverse(np.ones((3,4,1)))
    with pytest.raises(ValueError):reconstruct({},pair_count=0)
    f=np.array([[2.,3.,5.,7.]]);w=np.array([500.,501.,502.,503.])
    np.testing.assert_array_equal(spectral_bins(f,w,[500,502,504]),[[5.,12.]])
    with pytest.raises(ValueError):spectral_bins(f,w,[700,800])

def test_backend_unchanged_by_main_import():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before

def test_observation_sufficiency_and_metadata():
    from src.preprocessing import NOMINAL_PIXEL_UM,INTEGRATION_MS,PATTERN_PERIOD_US
    data=load_observations();assert set(data)=={'measurements','pattern_id','wavelength_nm'}
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in [str(NOMINAL_PIXEL_UM),str(INTEGRATION_MS),str(PATTERN_PERIOD_US)]:assert value in text
    for domain in ['github.com','githubusercontent.com','github.io','git.io']:assert domain not in text.lower()
