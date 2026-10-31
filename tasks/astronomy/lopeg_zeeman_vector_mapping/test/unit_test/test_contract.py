from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture(scope='module')
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return f['reconstruction'].copy()
def save(tmp_path,**data):
    path=tmp_path/'candidate.npz';np.savez_compressed(path,**data);return path
def test_reference_and_metric_known_cases(reference):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:cal=dict(f)
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:obs=dict(f)
    result=judge.score(reference,reference,cal,obs)
    assert result['observation_chi2_per_sample']<1.002
    for k,v in result.items():
        if k.endswith('_ncc'):assert v==pytest.approx(1,abs=1e-14)
        if k.endswith('_nrmse'):assert v==0
    x=np.array([1.,2.,3.]);y=np.array([1.,2.,4.])
    expected=np.sqrt(1/3)/3
    assert np.sqrt(np.mean((x-y)**2))/np.ptp(y)==pytest.approx(expected)
    assert np.vdot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))==pytest.approx(17/np.sqrt(294))
@pytest.mark.parametrize('kind',['missing_file','missing_key','shape','dtype','complex','nan','inf','object'])
def test_malformed_predictions(tmp_path,reference,kind):
    if kind=='missing_file':path=tmp_path/'absent.npz'
    elif kind=='missing_key':path=save(tmp_path,other=reference)
    else:
        x=reference.copy()
        if kind=='shape':x=x[...,1:]
        if kind=='dtype':x=x.astype(np.float32)
        if kind=='complex':x=x.astype(np.complex128)
        if kind in ('nan','inf'):x.flat[0]=np.nan if kind=='nan' else np.inf
        if kind=='object':x=np.array([object()],dtype=object)
        path=save(tmp_path,reconstruction=x)
    result,code=judge.evaluate(path)
    assert code==2 and result['status']=='error' and result['metrics']=={} and result['errors']
@pytest.mark.parametrize('kind',['uniform','scaled','shuffled','wrong_sign_component'])
def test_poor_spatial_results(tmp_path,reference,kind):
    x=reference.copy()
    if kind=='uniform':x[:]=1
    if kind=='scaled':x*=1.2
    if kind=='shuffled':x=x[...,::-1]
    if kind=='wrong_sign_component':
        if judge.MODE=='magnetic':x[:,1]*=-1
        else:x=2-x
    result,code=judge.evaluate(save(tmp_path,reconstruction=x))
    assert code==1 and result['status']=='fail' and not result['passed']
def test_foreign_cwd_cli_and_report_directory(tmp_path,reference):
    path=save(tmp_path,reconstruction=reference);report=tmp_path/'nested/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['passed']
def test_missing_evaluator_asset(tmp_path,reference):
    original=judge.ASSETS;judge.ASSETS=tmp_path
    try:
        result,code=judge.evaluate(save(tmp_path,reconstruction=reference));assert code==2 and result['metrics']=={} and 'Evaluator' in result['errors'][0]
    finally:judge.ASSETS=original
def test_report_write_failure(tmp_path,reference):
    path=save(tmp_path,reconstruction=reference);block=tmp_path/'block';block.write_text('file')
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(block/'report.json')],cwd=tmp_path,capture_output=True)
    assert run.returncode==2 and b'Cannot write report' in run.stderr
def test_zero_vector_prediction(tmp_path,reference):
    result,code=judge.evaluate(save(tmp_path,reconstruction=np.zeros_like(reference)))
    if judge.MODE=='magnetic':
        assert code==1 and result['status']=='fail'
        assert all(v==0 for k,v in result['metrics'].items() if k.endswith('_ncc'))
    else:assert code==2 and result['metrics']=={}
def test_metadata_and_visible_allowlist():
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        for key in f.files:assert '| '+key+' |' in readme and f[key].shape[0]==1
    for forbidden in ['github.com','github.io','raw.githubusercontent','baseline_reference.npz','ground_truth.npz']:
        assert forbidden not in readme
    for value in ['2456892.015','0.4232','65000','67.7','0.66','0.72','2.41','0.89','1.195','0.02']:
        assert value in readme
