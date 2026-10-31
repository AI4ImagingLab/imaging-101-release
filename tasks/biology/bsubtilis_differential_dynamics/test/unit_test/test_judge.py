import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def test_metric_known_case_all_quantities():
    ref=baseline();q=np.linspace(.75,4,39);p=ref['reconstruction'][0];t=judge.LAGS*.01
    ref['speed'][0]=np.exp(np.mean(np.log(p[:,:,2]/q),axis=1))
    obs=np.array([a[:,0,None]*(1-np.exp(-(a[:,2,None]*t)**a[:,3,None]))+a[:,1,None] for a in p])
    out=judge.score(ref,ref,obs,q);assert all(abs(v)<1e-14 for v in out.values())
    pred={k:v.copy() for k,v in ref.items()};pred['speed']*=1.1;out=judge.score(pred,ref,obs,q)
    for key,val in out.items():
        if 'observation' not in key:assert abs(val-.1)<1e-14

def test_invalid_contracts(tmp_path):
    ref=baseline();cases=[{},dict(ref,reconstruction=ref['reconstruction'].astype(np.float32)),dict(ref,speed=np.ones((2,))),dict(ref,reconstruction=ref['reconstruction'].astype(complex))]
    bad={k:v.copy() for k,v in ref.items()};bad['speed'][0,0]=np.nan;cases.append(bad)
    bad={k:v.copy() for k,v in ref.items()};bad['reconstruction'][0,0,0,3]=.9;cases.append(bad)
    for i,case in enumerate(cases):
        path=tmp_path/f'bad{i}.npz';np.savez(path,**case);report=tmp_path/f'r{i}.json'
        assert judge.evaluate(path,report)==2;r=json.loads(report.read_text());assert r['status']=='error' and r['metrics']=={} and r['errors']
    assert judge.evaluate(tmp_path/'missing',tmp_path/'missingreport.json')==2

def test_unavailable_reference_and_report_failure(tmp_path,monkeypatch):
    pred=tmp_path/'good.npz';np.savez(pred,**baseline());home=tmp_path/'hidden';home.mkdir();(home/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text());monkeypatch.setattr(judge,'HOME',home)
    assert judge.evaluate(pred,tmp_path/'error.json')==2
    assert judge.evaluate(pred,tmp_path)==2

def test_valid_poor_output(tmp_path):
    pred={k:v.copy() for k,v in baseline().items()};pred['reconstruction'][...,0]*=2;pred['speed']*=2
    p=tmp_path/'poor.npz';np.savez(p,**pred);report=tmp_path/'nested/report.json'
    assert judge.evaluate(p,report)==1;r=json.loads(report.read_text());assert r['status']=='fail' and not r['passed'] and len(r['metrics'])==5


def test_hidden_reference_does_not_define_target(tmp_path,monkeypatch):
    import os
    home=tmp_path/'evaluation';home.mkdir();(tmp_path/'data').mkdir()
    os.link(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    (home/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text())
    ref=baseline();truth={k:v.copy() for k,v in ref.items()};truth['reconstruction'][...,0]*=1.01
    np.savez(home/'baseline_reference.npz',**ref);np.savez(home/'ground_truth.npz',**truth)
    q=np.linspace(.75,4,39)
    truth['speed'][0]=np.exp(np.mean(np.log(truth['reconstruction'][0,:,:,2]/q),axis=1))
    pred=tmp_path/'truth.npz';np.savez(pred,**truth)
    t=judge.LAGS*.01;p=truth['reconstruction'][0]
    obs=np.array([a[:,0,None]*(1-np.exp(-(a[:,2,None]*t)**a[:,3,None]))+a[:,1,None] for a in p])
    monkeypatch.setattr(judge,'HOME',home);monkeypatch.setattr(judge,'direct_observation',lambda frames:(obs,q))
    assert judge.evaluate(pred,tmp_path/'truthreport.json')==0
    truth['speed'][0,0]=np.inf;np.savez(home/'ground_truth.npz',**truth)
    assert judge.evaluate(pred,tmp_path/'invalidreference.json')==0
