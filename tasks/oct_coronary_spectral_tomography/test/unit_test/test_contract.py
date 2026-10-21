from pathlib import Path
import importlib.util,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def test_known_metric():
    assert j.relative_l2(np.array([0,2j]),np.array([0,1j]))==1
    with pytest.raises(ValueError):j.relative_l2(np.ones(2),np.zeros(2))
@pytest.mark.parametrize('kind',['zero','shape','dtype','key','nan','inf','missing','object','overflow'])
def test_bad_prediction(tmp_path,kind):
    x=np.zeros(j.SHAPE,dtype=np.complex64);key='reconstruction'
    if kind=='shape':x=x[:,:,:1]
    if kind=='dtype':x=x.astype(np.complex128)
    if kind=='key':key='wrong'
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='object':x=np.array([object()],dtype=object)
    if kind=='overflow':x.fill(np.finfo(np.float32).max)
    path=tmp_path/'pred.npz'
    if kind!='missing':np.savez_compressed(path,**{key:x})
    report,code=j.evaluate(path)
    assert code==(1 if kind in ['zero','overflow'] else 2)
    assert report['status']==('fail' if code==1 else 'error')
    if code==2:assert report['metrics']=={} and report['errors']
def test_reference_and_foreign_cwd(tmp_path):
    report=tmp_path/'sub/report.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
def test_evaluator_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'ROOT',tmp_path)
    report,code=j.evaluate(tmp_path/'prediction.npz');assert code==2 and report['metrics']=={}
def test_report_write_failure(tmp_path):
    target=tmp_path/'block';target.write_text('file')
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(target/'a.json')],capture_output=True,text=True)
    assert p.returncode==2 and 'Cannot write' in p.stderr
