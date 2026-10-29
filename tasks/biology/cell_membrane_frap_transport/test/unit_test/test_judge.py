import importlib.util,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
from src.visualization import agreement

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics():
    a=np.array([0.,1.,2.]);b=np.array([0.,2.,4.])
    ncc,nrmse=agreement(a,b)
    assert ncc==pytest.approx(1.)
    assert nrmse==pytest.approx(np.sqrt(5/3)/4)
    p={'reconstruction':a,'diffusivity':np.array([[1.]])}
    r={'reconstruction':b,'diffusivity':np.array([[2.]])}
    m=judge.metrics(p,r)
    assert m['ncc']==pytest.approx(ncc) and m['nrmse']==pytest.approx(nrmse)
    assert m['diffusivity_relative_error']==.5
    assert agreement(a*0,b)[0]==0
    with pytest.raises(ValueError):agreement(a,np.ones(3))

@pytest.mark.parametrize('case',['pass','poor','missing','key','shape','dtype','complex','nan','inf'])
def test_judge_cli_contract(tmp_path,case):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:data={k:f[k].copy() for k in f.files}
    if case=='poor':data={k:np.zeros_like(v) for k,v in data.items()}
    if case=='key':del data['diffusivity']
    if case=='shape':data['reconstruction']=data['reconstruction'][0]
    if case=='dtype':data['reconstruction']=data['reconstruction'].astype(np.float32)
    if case=='complex':data['reconstruction']=data['reconstruction'].astype(complex)
    if case in ('nan','inf'):data['reconstruction'][0,0,0]=float(case)
    path=tmp_path/'pred.npz'
    if case!='missing':np.savez(path,**data)
    report=tmp_path/'nested/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    expected=0 if case=='pass' else 1 if case=='poor' else 2
    assert run.returncode==expected,run.stderr
    r=json.loads(report.read_text(encoding='utf8'))
    assert r['status']=={0:'pass',1:'fail',2:'error'}[expected]
    if expected==2: assert r['metrics']=={} and r['errors']

def test_missing_hidden_assets_and_write_failure(tmp_path):
    shutil.copy(ROOT/'evaluation/judge.py',tmp_path/'judge.py')
    shutil.copy(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    pred=ROOT/'evaluation/baseline_reference.npz'
    command=[sys.executable,str(tmp_path/'judge.py'),'--prediction',str(pred),'--report',str(tmp_path/'report.json')]
    assert subprocess.run(command,capture_output=True).returncode==2
    r=json.loads((tmp_path/'report.json').read_text(encoding='utf8'))
    assert r['metrics']=={} and r['errors']
    command[-1]=str(tmp_path)
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode==2 and 'Report write failed' in run.stderr
