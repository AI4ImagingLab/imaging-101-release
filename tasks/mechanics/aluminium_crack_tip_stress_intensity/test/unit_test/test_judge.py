from pathlib import Path
import importlib.util,subprocess,sys,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(tmp_path,d):
    p=tmp_path/'submission.npz';np.savez(p,**d);return p
def test_metrics_known_case():
    n,e=j.metrics(np.array([2.,4.]),np.array([1.,2.]));assert n==pytest.approx(1);assert e==pytest.approx(np.sqrt(2.5))
@pytest.mark.parametrize('kind',['missing','key','shape','int','complex','nan','inf'])
def test_malformed(tmp_path,kind):
    d=reference()
    if kind=='missing':p=tmp_path/'absent.npz'
    else:
        if kind=='key':del d['stress_intensity']
        elif kind=='shape':d['stress_intensity']=d['stress_intensity'][0]
        elif kind=='int':d['stress_intensity']=d['stress_intensity'].astype(int)
        elif kind=='complex':d['stress_intensity']=d['stress_intensity'].astype(complex)
        else:d['stress_intensity'][0,0]=np.nan if kind=='nan' else np.inf
        p=save(tmp_path,d)
    report,code=j.evaluate(p);assert code==2;assert report['metrics']=={};assert report['errors']
@pytest.mark.parametrize('key',list(j.SHAPES))
@pytest.mark.parametrize('kind',['zero','scale','negative','huge'])
def test_each_quantity_must_pass(tmp_path,key,kind):
    d=reference();d[key]=np.zeros_like(d[key]) if kind=='zero' else d[key]*({'scale':1.01,'negative':-1.,'huge':1e290}.get(kind,1))
    report,code=j.evaluate(save(tmp_path,d));assert code==1;assert not report['passed'];json.dumps(report,allow_nan=False)
def test_float32_and_foreign_cwd_cli(tmp_path):
    p=save(tmp_path,{k:v.astype('float32') for k,v in reference().items()});report=tmp_path/'nested/report.json'
    done=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert done.returncode==0,done.stderr;assert json.loads(report.read_text())['passed']
def test_missing_evaluator_asset(tmp_path):
    report,code=j.evaluate(save(tmp_path,reference()),tmp_path);assert code==2;assert report['metrics']=={}
def test_report_write_failure(tmp_path):
    p=save(tmp_path,reference());bad=tmp_path/'file';bad.write_text('file')
    done=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(bad/'child.json')],capture_output=True,text=True)
    assert done.returncode==2;assert 'Report write failed' in done.stderr
