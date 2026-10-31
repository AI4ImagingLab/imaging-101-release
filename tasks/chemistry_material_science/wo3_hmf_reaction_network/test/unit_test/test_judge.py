import importlib.util,json,sys,subprocess
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def reference():return judge.load(ROOT/'evaluation/baseline_reference.npz')

def test_independent_metric_case():
    b=reference();a={k:v.copy() for k,v in b.items()}
    delta=.001*np.ptp(b['reconstruction'][0,:,3]);a['reconstruction'][0,:,3]+=delta
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    metrics,_=judge.score(a,b,cfg)
    assert metrics['species_3_nrmse']==pytest.approx(.001,rel=1e-12)

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf','complex','poor','good'])
def test_native_contract(tmp_path,kind):
    a=reference();prediction=tmp_path/'p.npz';report=tmp_path/'nested/r.json'
    if kind=='key':a.pop('rates')
    if kind=='shape':a['rates']=a['rates'].ravel()
    if kind=='dtype':a['rates']=a['rates'].astype('float32')
    if kind=='complex':a['rates']=a['rates'].astype('complex128')
    if kind in ['nan','inf']:a['rates'][0,0]=float(kind)
    if kind=='poor':a={k:np.zeros_like(v) for k,v in a.items()}
    if kind!='missing':np.savez(prediction,**a)
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    expected=0 if kind=='good' else 1 if kind=='poor' else 2
    assert result.returncode==expected
    record=json.loads(report.read_text())
    assert record['passed']==(kind=='good')
    if expected==2:assert record['metrics']=={} and record['errors']

def test_evaluator_missing_asset_and_write_failure(tmp_path):
    import shutil
    folder=tmp_path/'eval';folder.mkdir();shutil.copy(ROOT/'evaluation/judge.py',folder/'judge.py');shutil.copy(ROOT/'evaluation/metrics.json',folder/'metrics.json')
    prediction=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'r.json'
    result=subprocess.run([sys.executable,str(folder/'judge.py'),'--prediction',str(prediction),'--report',str(report)],capture_output=True)
    assert result.returncode==2 and json.loads(report.read_text())['errors']
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(tmp_path)],capture_output=True)
    assert result.returncode==2 and b'Report-write failure' in result.stderr


def test_native_finite_prediction_metric_overflow_writes_error_json(tmp_path):
    """Finite valid-shape inputs must not prevent a machine-readable report."""
    arrays={key:np.full_like(value,1e300) for key,value in reference().items()}
    assert all(np.isfinite(value).all() for value in arrays.values())
    prediction=tmp_path/'finite_huge.npz';report=tmp_path/'nested/error.json'
    np.savez_compressed(prediction,**arrays)
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert result.returncode==2,result.stderr
    assert report.is_file()
    def reject_nonfinite(value):raise AssertionError('Nonfinite JSON number: '+value)
    record=json.loads(report.read_text(encoding='utf8'),parse_constant=reject_nonfinite)
    assert record['status']=='error' and record['passed'] is False
    assert record['metrics']=={} and record['thresholds']=={}
    assert record['errors'] and 'Nonfinite metric result' in record['errors'][0]
