from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def base():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in judge.SPECS}

def test_known_numerical_metrics():
    n,e=judge.metric_pair(np.array([2.,4.]),np.array([1.,3.]))
    assert n==pytest.approx(14/np.sqrt(200)) and e==pytest.approx(.5)
    assert judge.metric_pair(np.zeros(2),np.array([1.,3.]))[0]==0
    with pytest.raises(ValueError):judge.metric_pair(np.ones(2),np.ones(2))

def test_reference_passes_and_zero_prediction_fails(tmp_path):
    r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==0 and r['passed']
    d=base();d={k:np.zeros_like(v) for k,v in d.items()};p=tmp_path/'zero.npz';np.savez(p,**d)
    r,c=judge.evaluate(p);assert c==1 and r['status']=='fail' and r['metrics']

@pytest.mark.parametrize('fault',['missing_key','shape','int','float32','complex','nan','inf'])
def test_invalid_submissions_have_empty_metrics(tmp_path,fault):
    d=base();key='arterial_transit_time'
    if fault=='missing_key':del d[key]
    elif fault=='shape':d[key]=d[key][...,0]
    elif fault in ['int','float32','complex']:d[key]=d[key].astype({'int':'int64','float32':'float32','complex':'complex128'}[fault])
    else:d[key].flat[0]={'nan':np.nan,'inf':np.inf}[fault]
    p=tmp_path/'bad.npz';np.savez(p,**d);r,c=judge.evaluate(p)
    assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_missing_and_corrupt_npz(tmp_path):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    p=tmp_path/'corrupt.npz';p.write_text('not an archive');assert judge.evaluate(p)[1]==2

def test_missing_hidden_assets_and_invalid_configuration(tmp_path):
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)[1]==2
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['thresholds']['reconstruction_ncc']['boundary']=float('nan')
    (tmp_path/'metrics.json').write_text(json.dumps(config));r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert c==2 and r['metrics']=={}

def test_reference_precedence_and_bad_hidden_reference(tmp_path):
    assets=tmp_path/'evaluation';assets.mkdir();shutil.copy(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',assets/'baseline_reference.npz')
    np.savez(assets/'ground_truth.npz',reconstruction=np.ones((2,)))
    r,c=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets)
    assert c==2 and r['metrics']=={}

def test_cli_from_foreign_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')]
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['status']=='pass'
    blocked=tmp_path/'blocked';blocked.write_text('file');cmd[-1]=str(blocked/'report.json')
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==2 and 'Cannot write report' in p.stderr


def test_physical_nonnegativity_and_exact_exterior_zero(tmp_path):
    with np.load(ROOT/'data/raw_data.npz') as z:mask=z['mask']>0
    for key in judge.SPECS:
        for where,value in [(np.flatnonzero(~mask)[0],1e-12),(np.flatnonzero(mask)[0],-1e-12)]:
            d=base();d[key].flat[where]=value;p=tmp_path/'invalid_physical.npz';np.savez(p,**d)
            r,c=judge.evaluate(p)
            assert c==2 and r['metrics']=={},(key,where,value,r)
