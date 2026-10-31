from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('microstructure_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return {k:f[k].copy() for k in f.files}

def save(tmp_path,arrays):
    path=tmp_path/'prediction.npz';np.savez_compressed(path,**arrays);return path

def test_independent_known_metric_case():
    r=np.array([1.,2,4]);p=np.array([2.,4,8]);m=judge.metrics(p,r)
    assert m['ncc']==pytest.approx(1)
    assert m['nrmse']==pytest.approx(np.sqrt(7)/3)

def test_reference_and_float64_pass(tmp_path,reference):
    for dtype in [np.float32,np.float64]:
        report,code=judge.evaluate(save(tmp_path,{k:a.astype(dtype) for k,a in reference.items()}));assert code==0,report
        assert len(report['metrics'])==4

@pytest.mark.parametrize('key',judge.KEYS)
def test_each_scientific_quantity_independently_scored(tmp_path,reference,key):
    reference[key]*=0
    report,code=judge.evaluate(save(tmp_path,reference));assert code==1 and not report['passed']

@pytest.mark.parametrize('kind',['missing','extra','shape','int','complex','object','nan','inf','padding'])
def test_invalid_npz_has_empty_metrics(tmp_path,reference,kind):
    key=judge.KEYS[0]
    if kind=='missing':reference.pop(key)
    elif kind=='extra':reference['extra']=np.zeros(1)
    elif kind=='shape':reference[key]=reference[key][0]
    elif kind=='int':reference[key]=reference[key].astype(np.int64)
    elif kind=='complex':reference[key]=reference[key].astype(complex)
    elif kind=='object':reference[key]=reference[key].astype(object)
    elif kind=='nan':reference[key][0,44,11,1]=np.nan
    elif kind=='inf':reference[key][0,44,11,1]=np.inf
    elif kind=='padding':reference[key][0,0,0,0]=1
    else:reference={k:a.astype(float)*1e300 for k,a in reference.items()}
    report,code=judge.evaluate(save(tmp_path,reference));assert code==2 and report['metrics']=={} and report['errors'],report
    json.dumps(report,allow_nan=False)

def test_missing_file(tmp_path):
    report,code=judge.evaluate(tmp_path/'absent.npz');assert code==2 and not report['metrics']

def test_missing_evaluator_and_invalid_threshold(tmp_path,reference,monkeypatch):
    prediction=save(tmp_path,reference);monkeypatch.setattr(judge,'REFERENCE',tmp_path/'absent.npz')
    report,code=judge.evaluate(prediction);assert code==2
    monkeypatch.setattr(judge,'REFERENCE',ROOT/'evaluation/baseline_reference.npz')
    config=json.loads(judge.CONFIG.read_text());config['maximum_normalized_mse_excess_boundary']=-1
    path=tmp_path/'metrics.json';path.write_text(json.dumps(config));monkeypatch.setattr(judge,'CONFIG',path)
    report,code=judge.evaluate(prediction);assert code==2 and report['metrics']=={}

def test_foreign_cwd_and_report_write_error(tmp_path,reference):
    prediction=save(tmp_path,reference);report=tmp_path/'nested'/'report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,encoding="utf8",errors="replace");assert run.returncode==0,run.stderr
    assert json.loads(report.read_text())['passed']
    run=subprocess.run(cmd[:-1]+[str(prediction/'bad.json')],cwd=tmp_path,capture_output=True,text=True,encoding="utf8",errors="replace");assert run.returncode==2 and 'Cannot write Judge report' in run.stderr
