import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def data():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as a:return {k:a[k].copy() for k in a.files}

def test_known_metrics(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**data());r,c=judge.evaluate(p);assert c==0;assert r['metrics']['parameter_relative_max']==0;assert r['metrics']['curve_relative_l2']==0

def test_bad_quality(tmp_path):
    d=data();d['reconstruction']*=2;d['correlation']*=.5;p=tmp_path/'p.npz';np.savez(p,**d);r,c=judge.evaluate(p);assert c==1 and r['status']=='fail'

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','corrupt'])
def test_malformed(tmp_path,kind):
    d=data();p=tmp_path/'p.npz'
    if kind=='key':del d['reconstruction']
    if kind=='shape':d['correlation']=d['correlation'].T
    if kind=='dtype':d['reconstruction']=d['reconstruction'].astype('float32')
    if kind=='complex':d['reconstruction']=d['reconstruction'].astype('complex128')
    if kind in ['nan','inf']:d['reconstruction'][0,0]=float(kind)
    if kind=='corrupt':p.write_text('bad')
    elif kind!='missing':np.savez(p,**d)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

def test_missing_evaluator(tmp_path):
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2 and r['metrics']=={}


def test_independent_known_parameter_error(tmp_path):
    d=data();d['reconstruction'][0,0]*=1.1;p=tmp_path/'p.npz';np.savez(p,**d)
    r,c=judge.evaluate(p);assert c==1;assert abs(r['metrics']['parameter_relative_max']-.1)<1e-14
    assert r['metrics']['curve_relative_l2']==0

@pytest.mark.parametrize('fault',['zero','nan','dtype','boundary'])
def test_bad_evaluator_assets(tmp_path,fault):
    import json
    d=data();config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if fault=='zero':d['reconstruction'][0,1]=0
    if fault=='nan':d['correlation'][0,0]=np.nan
    if fault=='dtype':d['correlation']=d['correlation'].astype('float32')
    if fault=='boundary':config['curve_relative_l2_boundary']=-1
    np.savez(tmp_path/'baseline_reference.npz',**d)
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf-8')
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2 and r['metrics']=={}

def test_cli_foreign_working_directory_and_report_failure(tmp_path):
    import subprocess,sys,json
    dest=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(dest)],cwd=tmp_path,capture_output=True)
    assert r.returncode==0 and json.loads(dest.read_text())['passed']
    blocker=tmp_path/'file';blocker.write_text('block')
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(blocker/'report.json')],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and b'Report write failed' in r.stderr


@pytest.mark.parametrize('values',[[100001.,.2,.03],[5.,.2,1001.],[1e308,.2,1e308]])
def test_upper_domain_bounds(tmp_path,values):
    d=data();d['reconstruction'][0]=values;p=tmp_path/'p.npz';np.savez(p,**d);r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_actual_truth_precedence(tmp_path):
    import shutil
    base=tmp_path/'evaluation';base.mkdir();shutil.copy2(ROOT/'evaluation/metrics.json',base/'metrics.json');shutil.copytree(ROOT/'data',tmp_path/'data')
    truth=data();bad=data();bad['reconstruction'][0,0]*=2
    np.savez(base/'ground_truth.npz',**truth);np.savez(base/'baseline_reference.npz',**bad)
    p=tmp_path/'p.npz';np.savez(p,**truth);r,c=judge.evaluate(p,base);assert c==0 and r['metrics']['parameter_relative_max']==0

@pytest.mark.parametrize('value',[100001.,-1.,np.inf])
def test_nonphysical_hidden_reference(tmp_path,value):
    import shutil
    shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json');d=data();d['reconstruction'][0,0]=value;np.savez(tmp_path/'baseline_reference.npz',**d)
    r,c=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert c==2 and r['metrics']=={}
