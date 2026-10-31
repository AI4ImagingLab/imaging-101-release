import copy,json,subprocess,sys,tempfile,shutil
from pathlib import Path
import numpy as np
import pytest
from evaluation.judge import evaluate,HERE
@pytest.fixture
def good():
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def test_reference_and_poor(good):
    assert evaluate(good)['passed']
    bad=copy.deepcopy(good);bad['reconstruction']+=1
    assert evaluate(bad)['status']=='fail'
@pytest.mark.parametrize('fault',['missing','shape','dtype','nan','inf','covariance','zero_reference','bad_boundary'])
def test_malformed_and_evaluator(good,fault):
    bad=copy.deepcopy(good);kwargs={}
    if fault=='missing':bad.pop('reconstruction')
    if fault=='shape':bad['reconstruction']=bad['reconstruction'][0]
    if fault=='dtype':bad['reconstruction']=bad['reconstruction'].astype('float32')
    if fault in ['nan','inf']:bad['hubble_rate'][0,0,0]=float(fault)
    if fault=='covariance':bad['point_covariance'][0,0,0,0,0]=-1
    if fault=='zero_reference':
        reference=copy.deepcopy(good);reference['hubble_rate']*=0;kwargs['reference']=reference
    if fault=='bad_boundary':
        config=json.loads((HERE/'metrics.json').read_text());config['maximum_q_absolute_error_boundary']=float('inf');kwargs['configuration']=config
    report=evaluate(bad,**kwargs)
    assert report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)
def test_cli_missing_prediction_and_report_write():
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as directory:
        _check_cli(Path(directory))
def _check_cli(tmp_path):
    report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(HERE/'judge.py'),'--prediction',str(tmp_path/'absent.npz'),'--report',str(report)]
    result=subprocess.run(cmd,capture_output=True)
    assert result.returncode==2 and json.loads(report.read_text())['status']=='error'
    cmd[-1]=str(tmp_path)
    result=subprocess.run(cmd,capture_output=True)
    assert result.returncode==2 and b'report write failed' in result.stderr
def test_cwd_independence_and_missing_hidden_assets():
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as directory:
        work=Path(directory);report=work/'report.json'
        command=[sys.executable,str(HERE/'judge.py'),'--prediction',str(HERE.parent/'output/reconstruction.npz'),'--report',str(report)]
        result=subprocess.run(command,cwd=work,capture_output=True)
        assert result.returncode==0 and json.loads(report.read_text())['passed']
        shutil.copyfile(HERE/'judge.py',work/'judge.py');command[1]=str(work/'judge.py')
        result=subprocess.run(command,cwd=work,capture_output=True)
        r=json.loads(report.read_text());assert result.returncode==2 and r['status']=='error' and r['metrics']=={}

def test_actual_truth_precedence(good,monkeypatch,tmp_path):
    import evaluation.judge as module
    target=tmp_path/'evaluation';target.mkdir()
    shutil.copyfile(HERE/'metrics.json',target/'metrics.json')
    np.savez(target/'baseline_reference.npz',**good)
    truth=copy.deepcopy(good);truth['reconstruction']+=.01
    np.savez(target/'ground_truth.npz',**truth)
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
    monkeypatch.setattr(module,'HERE',target)
    r=module.evaluate(good,observations=observations)
    assert r['status']=='fail' and r['metrics']['maximum_q_absolute_error']==pytest.approx(.01)
