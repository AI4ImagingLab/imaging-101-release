from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_known_metrics():
    from src.visualization import metrics as visual_metrics
    r=np.array([1.,2.,3.]);x=np.array([3.,2.,1.])
    m=j.agreement(x,r)
    assert m['ncc']==pytest.approx(10/14)
    assert m['nrmse']==pytest.approx(np.sqrt(8/3)/2)
    assert visual_metrics(x,r)==m
    assert j.agreement(np.zeros(3),r)['ncc']==0
    with pytest.raises(ValueError):j.agreement(x,np.ones(3))

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','broken'])
def test_malformed(tmp_path,kind):
    path=tmp_path/'input.npz'
    x=np.zeros(j.SHAPE)
    if kind=='missing':pass
    elif kind=='key':np.savez(path,wrong=x)
    elif kind=='shape':np.savez(path,reconstruction=np.zeros((7,21)))
    elif kind=='dtype':np.savez(path,reconstruction=x.astype(int))
    elif kind=='complex':np.savez(path,reconstruction=x.astype(complex))
    elif kind=='broken':path.write_bytes(b'not an npz')
    else:
        x.flat[0]=np.nan if kind=='nan' else np.inf
        np.savez(path,reconstruction=x)
    report=j.evaluate(path)
    assert report['status']=='error' and report['metrics']=={} and report['errors']

def test_quality_and_assets(tmp_path):
    source=ROOT/'evaluation/baseline_reference.npz'
    assert j.evaluate(source)['status']=='pass'
    r=np.load(source)['reconstruction']
    for x in [np.zeros_like(r),2*r]:
        path=tmp_path/'poor.npz';np.savez(path,reconstruction=x)
        assert j.evaluate(path)['status']=='fail'
    assert j.evaluate(source,tmp_path)['status']=='error'

def test_cli_contract(tmp_path):
    path=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(path)]
    run=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads(path.read_text(encoding='utf8'));assert report['passed']
    command[3]=str(tmp_path/'missing.npz')
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
    command[3]=str(ROOT/'evaluation/baseline_reference.npz');command[-1]=str(tmp_path)
    run=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==2 and 'report write failed' in run.stderr
