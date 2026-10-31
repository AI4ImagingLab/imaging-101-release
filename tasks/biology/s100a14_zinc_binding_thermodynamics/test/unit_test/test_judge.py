from pathlib import Path
import subprocess,sys,importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def test_known_metric():
    r={k:np.ones(v) for k,v in j.SHAPES.items()};p={k:a+2 for k,a in r.items()}
    v,passed=j.score(p,r,np.ones((1,6,14)),{k:1 for k in list(r)+['observation_rmse']})
    assert all(x==2 for x in v.values()) and not passed
@pytest.mark.parametrize('kind',['missing_key','shape','dtype','nan','inf','complex','absent'])
def test_bad_inputs(kind,tmp_path):
    r=j.load_arrays(ROOT/'evaluation/baseline_reference.npz');p=tmp_path/'bad.npz'
    if kind=='missing_key':r.pop('enthalpy')
    if kind=='shape':r['enthalpy']=np.zeros(2)
    if kind=='dtype':r['enthalpy']=r['enthalpy'].astype('float32')
    if kind=='complex':r['enthalpy']=r['enthalpy'].astype('complex128')
    if kind=='nan':r['enthalpy'][0,0]=np.nan
    if kind=='inf':r['enthalpy'][0,0]=np.inf
    if kind!='absent':np.savez(p,**r)
    report,code=j.evaluate(p)
    assert code==2 and report['metrics']=={} and report['errors']
def test_poor_and_good(tmp_path):
    p=tmp_path/'zero.npz';np.savez(p,**{k:np.zeros(v) for k,v in j.SHAPES.items()})
    r,c=j.evaluate(p);assert c==1 and r['status']=='fail'
    r,c=j.evaluate(ROOT/'output/reconstruction.npz');assert c==0 and r['passed']
def test_cwd_and_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==0
    command[-1]=str(tmp_path)
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
def test_evaluator_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'HERE',tmp_path)
    r,c=j.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and not r['metrics']
def test_truth_precedence_and_missing_reference(tmp_path,monkeypatch):
    import shutil
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'evaluation/metrics.json',e/'metrics.json')
    r=j.load_arrays(ROOT/'evaluation/baseline_reference.npz')
    np.savez(tmp_path/'data/raw_data.npz',heats=r['predicted_heats'])
    np.savez(e/'baseline_reference.npz',**r)
    altered={k:a+100 for k,a in r.items()};np.savez(e/'ground_truth.npz',**altered)
    monkeypatch.setattr(j,'HERE',e)
    report,c=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==1
    assert report['metrics']['enthalpy']==100
def test_non_npz(tmp_path):
    p=tmp_path/'bad.npz';p.write_text('invalid',encoding='utf8')
    r,c=j.evaluate(p);assert c==2 and not r['metrics']
