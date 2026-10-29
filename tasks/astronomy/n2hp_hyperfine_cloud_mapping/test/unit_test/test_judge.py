import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as a:return {k:a[k] for k in a.files}

@pytest.mark.parametrize('defect',['missing_key','wrong_shape','integer','complex','nan','inf','negative_width','object'])
def test_reject_malformed(tmp_path,defect):
    d=valid();k='dispersion'
    if defect=='missing_key':del d[k]
    if defect=='wrong_shape':d[k]=d[k][0]
    if defect=='integer':d[k]=d[k].astype('int64')
    if defect=='complex':d[k]=d[k].astype('complex128')
    if defect=='nan':d[k][0,0,0]=np.nan
    if defect=='inf':d[k][0,0,0]=np.inf
    if defect=='negative_width':d[k]=-d[k]
    if defect=='object':d[k]=d[k].astype(object)
    p=tmp_path/'bad.npz';np.savez(p,**d)
    report,code=judge.evaluate(p)
    assert code==2;assert report['status']=='error';assert report['metrics']=={};assert report['errors']

def test_missing_file_and_broken_npz(tmp_path):
    for p in [tmp_path/'missing.npz',tmp_path/'broken.npz']:
        if p.name=='broken.npz':p.write_text('not an npz')
        report,code=judge.evaluate(p);assert code==2;assert report['metrics']=={}

def test_poor_but_valid_output_fails(tmp_path):
    d=valid();d['velocity']+=.1
    p=tmp_path/'poor.npz';np.savez(p,**d)
    r,code=judge.evaluate(p);assert code==1;assert r['status']=='fail';assert not r['passed']

def test_valid_float32_and_reference_pass(tmp_path):
    for dtype in ['float64','float32']:
        p=tmp_path/(dtype+'.npz');np.savez(p,**{k:v.astype(dtype) for k,v in valid().items()})
        r,code=judge.evaluate(p);assert code==0;assert r['passed']

def test_hidden_asset_failures_and_invalid_threshold(tmp_path):
    e=tmp_path/'evaluation';e.mkdir()
    shutil.copy(ROOT/'evaluation/metrics.json',e/'metrics.json')
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',e);assert c==2;assert r['metrics']=={}
    config=json.loads((e/'metrics.json').read_text());config['parameter_relative_max_boundary']=0
    (e/'metrics.json').write_text(json.dumps(config))
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',e);assert c==2;assert r['metrics']=={}

def test_cli_foreign_cwd_and_report_write_failure(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';target=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(target)]
    r=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0;assert json.loads(target.read_text())['passed']
    command[-1]=str(tmp_path)
    r=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2;assert 'Could not write report' in r.stderr

def test_ground_truth_precedence(tmp_path):
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir()
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/name,e/name)
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    d=valid();d['velocity']+=.1;np.savez(e/'ground_truth.npz',**d)
    r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',e)
    assert c==1;assert r['metrics']['velocity_relative_max']>.01
