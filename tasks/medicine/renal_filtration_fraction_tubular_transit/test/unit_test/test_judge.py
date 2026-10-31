from pathlib import Path
import importlib.util,subprocess,sys,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def reference():return judge.load_arrays(ROOT/'evaluation/baseline_reference.npz')
def test_reference_and_metric_case(reference):
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==0
    changed={k:v.copy() for k,v in reference.items()};changed['reconstruction'][0,:,0]+=.05
    assert abs(judge.metrics(changed,reference)['flow_scaled_rmse']-1)<1e-14
@pytest.mark.parametrize('kind',['missing_key','shape','dtype','complex','nan','inf'])
def test_invalid(kind,reference,tmp_path):
    a={k:v.copy() for k,v in reference.items()}
    if kind=='missing_key':a.pop('signal')
    elif kind=='shape':a['signal']=a['signal'][0]
    elif kind=='dtype':a['signal']=a['signal'].astype(np.float32)
    elif kind=='complex':a['signal']=a['signal'].astype(complex)
    else:a['signal'].flat[0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'invalid.npz';np.savez(p,**a);r,c=judge.evaluate(p)
    assert c==2 and r['metrics']=={} and r['errors']
def test_poor_each_parameter_and_signal(reference,tmp_path):
    for j in range(4):
        a={k:v.copy() for k,v in reference.items()};a['reconstruction'][0,:,j]*=2
        p=tmp_path/f'bad{j}.npz';np.savez(p,**a);assert judge.evaluate(p)[1] in (1,2)
    a={k:v.copy() for k,v in reference.items()};a['signal']*=0
    p=tmp_path/'signal.npz';np.savez(p,**a);assert judge.evaluate(p)[1]==1
def test_missing_assets_config(tmp_path,monkeypatch):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    monkeypatch.setattr(judge,'HERE',tmp_path);assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
    (tmp_path/'metrics.json').write_text('{"thresholds":{}}',encoding='utf8');assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
def test_cli_foreign_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'reports/result.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'reports/result.json').read_text())['passed']
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
