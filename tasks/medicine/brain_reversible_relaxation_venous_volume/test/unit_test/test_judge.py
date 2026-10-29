from pathlib import Path
import importlib.util,sys,subprocess,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():return judge.load_arrays(ROOT/'evaluation/baseline_reference.npz')

def test_reference_and_metric_case(reference):
    result,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==0 and result['passed']
    mask=judge.observation_mask();modified={k:v.copy() for k,v in reference.items()}
    modified['reconstruction'][0][mask]+=2
    measured=judge.metrics(modified,reference)
    for c in range(4):assert abs(measured[f'r2prime_rmse_{c}']-2)<1e-14

@pytest.mark.parametrize('kind',['missing_key','shape','dtype','nan','inf','complex'])
def test_invalid_inputs(kind,reference,tmp_path):
    a={k:v.copy() for k,v in reference.items()}
    if kind=='missing_key':a.pop('dbv')
    elif kind=='shape':a['dbv']=a['dbv'][0]
    elif kind=='dtype':a['dbv']=a['dbv'].astype(np.float32)
    elif kind=='complex':a['dbv']=a['dbv'].astype(complex)
    else:a['dbv'].flat[0]=np.nan if kind=='nan' else np.inf
    path=tmp_path/'bad.npz';np.savez(path,**a)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_poor_outputs_each_quantity(reference,tmp_path):
    for key in reference:
        a={k:v.copy() for k,v in reference.items()};a[key]*=0
        path=tmp_path/f'{key}.npz';np.savez(path,**a)
        report,code=judge.evaluate(path);assert code==1 and not report['passed']
    a={k:v.copy() for k,v in reference.items()};a['dbv'].flat[0]=1
    np.savez(tmp_path/'outside.npz',**a)
    assert judge.evaluate(tmp_path/'outside.npz')[1]==1

def test_missing_assets_and_config(tmp_path,monkeypatch):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    monkeypatch.setattr(judge,'HERE',tmp_path)
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
    (tmp_path/'metrics.json').write_text('{"thresholds":{}}',encoding='utf8')
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2

def test_cli_foreign_cwd_and_report_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['status']=='pass'
    command[-1]=str(tmp_path)
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
