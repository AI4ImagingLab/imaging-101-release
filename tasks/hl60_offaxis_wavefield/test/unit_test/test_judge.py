from pathlib import Path
import importlib.util,sys,subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def references():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def test_known_metric():
    assert j.relative_l2(np.array([0,2]),np.array([0,1]))==1
    assert j.relative_l2(np.array([0,2j]),np.array([0,1j]))==1
    with pytest.raises(ValueError):j.relative_l2(np.ones(2),np.zeros(2))
@pytest.mark.parametrize('kind',['zero','missing','key','shape','dtype','nan','inf','object','overflow'])
def test_prediction_errors(tmp_path,kind):
    arrays=references();key=next(iter(arrays));x=arrays[key]
    if kind=='zero':arrays={k:np.zeros_like(v) for k,v in arrays.items()}
    if kind=='key':del arrays[key]
    if kind=='shape':arrays[key]=x[...,0]
    if kind=='dtype':arrays[key]=x.astype('complex64' if np.iscomplexobj(x) else 'float32')
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='object':arrays[key]=np.array([object()],dtype=object)
    if kind=='overflow':x.fill(np.finfo(np.float64).max)
    path=tmp_path/'input.npz'
    if kind!='missing':np.savez_compressed(path,**arrays)
    report,code=j.evaluate(path)
    assert code==(1 if kind=='zero' else 2)
    if code==2:assert report['metrics']=={} and report['errors']
def test_every_output_judged(tmp_path):
    for key in references():
        arrays=references();arrays[key]*=0
        path=tmp_path/'one_bad.npz';np.savez_compressed(path,**arrays)
        report,code=j.evaluate(path);assert code==1 and not report['passed']
def test_foreign_cwd_and_write_error(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'sub/report.json')]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    (tmp_path/'blocked').write_text('file');command[-1]=str(tmp_path/'blocked/report.json')
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==2 and 'Cannot write' in result.stderr
def test_evaluator_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'ROOT',tmp_path);r,c=j.evaluate('missing.npz');assert c==2 and r['metrics']=={}
