from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("judge",ROOT/"evaluation/judge.py");j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def baseline():
    with np.load(j.REFERENCE,allow_pickle=False) as z:return {k:z[k].copy() for k in j.KEYS}
def write(tmp_path,fields):
    p=tmp_path/"prediction.npz";np.savez(p,**fields);return p
def test_original_passes(tmp_path):
    r,code=j.evaluate(write(tmp_path,baseline()));assert code==0 and r['passed'];assert len(r['metrics'])==2
@pytest.mark.parametrize('kind',['missingkey','extrakey','shape','int','complex','nan','inf','negative'])
def test_malformed_contract(tmp_path,kind):
    x=baseline();k=j.KEYS[0]
    if kind=='missingkey':x.pop(k)
    elif kind=='extrakey':x['extra']=np.zeros(1)
    elif kind=='shape':x[k]=x[k][0]
    elif kind=='int':x[k]=x[k].astype(np.int64)
    elif kind=='complex':x[k]=x[k].astype(np.complex128)
    elif kind=='negative':x[k][0,0,0]=-1
    else:x[k][0,0,0]=np.nan if kind=='nan' else np.inf
    r,code=j.evaluate(write(tmp_path,x));assert code==2 and not r['metrics'] and r['errors']
def test_missing_file(tmp_path):
    r,code=j.evaluate(tmp_path/'missing.npz');assert code==2 and not r['metrics']
def test_zero_and_uniform_controls_fail(tmp_path):
    for val in [0.,.05]:
        x={k:np.full(j.SHAPE,val) for k in j.KEYS};r,code=j.evaluate(write(tmp_path,x));assert code in (1,2) and not r['passed']
def test_every_type_is_scored(tmp_path):
    for k in range(20):
        x=baseline();x['cell_type_proportions'][0,:,k]*=.8
        r,code=j.evaluate(write(tmp_path,x));assert code==1;assert r['metrics']['mixture_relation_error']>1e-6
@pytest.mark.parametrize('kind',['absent','nan','shape','constant'])
def test_bad_evaluator_assets(tmp_path,monkeypatch,kind):
    pred=write(tmp_path,baseline());bad=tmp_path/'badref.npz';x=baseline()
    if kind=='nan':x[j.KEYS[0]][0,0,0]=np.nan
    if kind=='shape':x[j.KEYS[0]]=x[j.KEYS[0]][0]
    if kind=='constant':x={k:np.ones(j.SHAPE) for k in j.KEYS}
    if kind!='absent':np.savez(bad,**x)
    monkeypatch.setattr(j,'REFERENCE',bad);r,code=j.evaluate(pred);assert code==2 and not r['metrics']
def test_actual_truth_precedence_and_invalid_truth(tmp_path,monkeypatch):
    x=np.arange(428,dtype=float)[:,None];k=np.arange(20,dtype=float)[None,:]
    V=.02+(x+1)*(k+1)/100000;P=V/V.sum(1,keepdims=True)
    truth={'mixing_weights':V[None],'cell_type_proportions':P[None]}
    gt=tmp_path/'ground_truth.npz';np.savez(gt,**truth);monkeypatch.setattr(j,'TRUTH',gt)
    r,c=j.evaluate(write(tmp_path,truth));assert c==2 and 'Unexpected actual truth' in r['errors'][0]
    r,c=j.evaluate(write(tmp_path,baseline()));assert c==2
    truth['mixing_weights'][0,0,0]=np.nan;np.savez(gt,**truth)
    r,c=j.evaluate(write(tmp_path,baseline()));assert c==2 and not r['metrics']
def test_metric_analytic_independent():
    r=np.broadcast_to(np.arange(428.)[None,:,None]+1,(1,428,20)).copy();x=2*r
    out=j.field_metrics(x,r)
    assert out['type00_ncc']==pytest.approx(1.,abs=2e-15)
    assert out['type00_nrmse']==pytest.approx(np.sqrt(np.mean(np.arange(1,429.)**2))/427)
    assert out['type00_max_absolute_error']==428
def test_invalid_threshold(tmp_path,monkeypatch):
    c=json.loads(j.METRICS.read_text());c['profile_nll_excess_boundary']=float('nan');path=tmp_path/'metrics.json';path.write_text(json.dumps(c));monkeypatch.setattr(j,'METRICS',path)
    r,code=j.evaluate(write(tmp_path,baseline()));assert code==2 and not r['metrics']
def test_foreign_cwd_and_report_write_failure(tmp_path):
    p=write(tmp_path,baseline());script=ROOT/'evaluation/judge.py';report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,str(script),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert r.returncode==0 and json.loads(report.read_text())['passed']
    blocker=tmp_path/'blocker';blocker.write_text('file')
    r=subprocess.run([sys.executable,str(script),'--prediction',str(p),'--report',str(blocker/'report.json')],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and b'Cannot write judge report' in r.stderr
