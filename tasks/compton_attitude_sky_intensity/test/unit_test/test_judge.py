from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_compton_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as data:return data['reconstruction']

def prediction(tmp_path,values):
    path=tmp_path/'prediction.npz';np.savez_compressed(path,reconstruction=values);return path

def test_known_good_metric_case(tmp_path,reference):
    report,code=judge.evaluate(prediction(tmp_path,reference));assert code==0 and report['passed']
    assert report['metrics']['ncc']==pytest.approx(1) and report['metrics']['nrmse']==0

@pytest.mark.parametrize('case',['zero','flat','scale','permute'])
def test_reject_physical_or_image_failure(tmp_path,reference,case):
    fields=dict(zero=np.zeros_like(reference),flat=np.full_like(reference,reference.mean()),scale=reference*.95,permute=np.roll(reference,307))
    report,code=judge.evaluate(prediction(tmp_path,fields[case]));assert code==1 and report['status']=='fail' and report['metrics']

@pytest.mark.parametrize('case',['shape','integer','complex','float16','nan','inf','negative','key','missing'])
def test_invalid_submission(tmp_path,reference,case):
    values=dict(shape=reference.ravel(),integer=reference.astype(np.int64),complex=reference.astype(np.complex64),float16=reference.astype(np.float16),nan=reference.copy(),inf=reference.copy(),negative=-reference)
    if case=='missing':path=tmp_path/'absent.npz'
    elif case=='key':path=tmp_path/'badkey.npz';np.savez_compressed(path,wrong=reference)
    else:
        array=values[case]
        if case=='nan':array[0,0]=np.nan
        if case=='inf':array[0,0]=np.inf
        path=prediction(tmp_path,array)
    report,code=judge.evaluate(path);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('value',[True,'0.001',float('nan'),0,-1])
def test_invalid_numeric_configuration(tmp_path,reference,value):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['nrmse_boundary']=value
    directory=tmp_path/'evaluation';directory.mkdir();(directory/'metrics.json').write_text(json.dumps(config))
    np.savez_compressed(directory/'baseline_reference.npz',reconstruction=reference)
    report,code=judge.evaluate(prediction(tmp_path,reference),directory,ROOT/'data/raw_data.npz');assert code==2 and report['metrics']=={}

def test_truth_priority_and_missing_assets(tmp_path,reference):
    directory=tmp_path/'evaluation';directory.mkdir();(directory/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    path=prediction(tmp_path,reference)
    report,code=judge.evaluate(path,directory,ROOT/'data/raw_data.npz');assert code==2
    np.savez_compressed(directory/'baseline_reference.npz',reconstruction=np.roll(reference,307))
    np.savez_compressed(directory/'ground_truth.npz',reconstruction=reference)
    report,code=judge.evaluate(path,directory,ROOT/'data/raw_data.npz');assert code==0 and report['metrics']['nrmse']==0
    (directory/'ground_truth.npz').unlink()
    report,code=judge.evaluate(path,directory,ROOT/'data/raw_data.npz');assert code==1 and report['metrics']['nrmse']>0

def test_foreign_cwd_and_report_failure(tmp_path,reference):
    path=prediction(tmp_path,reference);destination=tmp_path/'nested/report.json'
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(destination)],cwd=tmp_path,capture_output=True)
    assert child.returncode==0 and json.loads(destination.read_text())['passed']
    blocked=tmp_path/'blocked';blocked.write_text('file blocks directory')
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True)
    assert child.returncode==2 and b'Cannot write judge report' in child.stderr


def test_existing_truth_directory_is_evaluator_error(tmp_path,reference):
    directory=tmp_path/'evaluation';directory.mkdir()
    (directory/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    np.savez_compressed(directory/'baseline_reference.npz',reconstruction=reference)
    (directory/'ground_truth.npz').mkdir()
    report,code=judge.evaluate(prediction(tmp_path,reference),directory,ROOT/'data/raw_data.npz')
    assert code==2 and report['status']=='error'
    assert report['metrics']=={} and report['errors']


def test_finite_extreme_prediction_has_structured_error_report(tmp_path):
    path=prediction(tmp_path,np.full((1,3072),1e308,dtype=np.float64))
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)
    destination=tmp_path/'extreme_report.json'
    child=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(destination)],cwd=tmp_path,capture_output=True)
    assert child.returncode==2
    parsed=json.loads(destination.read_text(encoding='utf-8'))
    assert parsed['status']=='error' and parsed['passed'] is False and parsed['metrics']=={} and parsed['errors']
    json.dumps(parsed,allow_nan=False)
