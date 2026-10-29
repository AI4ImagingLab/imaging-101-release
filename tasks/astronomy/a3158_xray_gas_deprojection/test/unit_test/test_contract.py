import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.solvers import median_smooth,reconstruct
from src.preprocessing import load_data

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def sandbox(tmp_path,monkeypatch):
    target=tmp_path/'evaluation';target.mkdir()
    for name in ('metrics.json','baseline_reference.npz'):
        (target/name).write_bytes((ROOT/'evaluation'/name).read_bytes())
    # Keep independent observation calibration visible at its documented path.
    (tmp_path/'data').mkdir();(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes())
    monkeypatch.setattr(judge,'HERE',target)
    return target

def test_known_metrics_and_median_endpoints():
    ncc,nrmse=judge.compare(np.array([0.,2.]),np.array([0.,1.]))
    assert_allclose([ncc,nrmse],[1.,np.sqrt(.5)])
    assert_allclose(median_smooth(np.arange(7.)),[0.,1.5,2.,3.,4.,4.5,6.])

def test_fresh_pass_and_both_quantities_consumed(tmp_path):
    out,_=reconstruct(load_data());path=tmp_path/'prediction.npz';np.savez(path,**out)
    result=judge.evaluate(path);assert result['passed'];assert len(result['metrics'])==3
    for key in out:
        bad=dict(out);bad[key]=np.zeros_like(bad[key]);np.savez(path,**bad)
        assert not judge.evaluate(path)['passed']

@pytest.mark.parametrize('mutation',['missing','shape','dtype','nan','inf','complex'])
def test_invalid_submissions(tmp_path,mutation):
    out,_=reconstruct(load_data())
    if mutation=='missing':out.pop('emissivity')
    elif mutation=='shape':out['reconstruction']=out['reconstruction'][0]
    elif mutation=='dtype':out['reconstruction']=out['reconstruction'].astype('float32')
    elif mutation=='complex':out['reconstruction']=out['reconstruction'].astype('complex128')
    else:out['reconstruction'][0,0]=np.nan if mutation=='nan' else np.inf
    path=tmp_path/'p.npz';np.savez(path,**out)
    with pytest.raises((ValueError,KeyError)):judge.evaluate(path)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,None,'0.1',True])
def test_invalid_configuration(tmp_path,monkeypatch,value):
    target=sandbox(tmp_path,monkeypatch);config=json.loads((target/'metrics.json').read_text())
    if value is None:config.pop('density_observation_rms_boundary')
    else:config['density_observation_rms_boundary']=value
    (target/'metrics.json').write_text(json.dumps(config))
    with pytest.raises((ValueError,KeyError)):judge.evaluate(target/'baseline_reference.npz')

def test_every_declared_threshold_is_consumed(tmp_path,monkeypatch):
    target=sandbox(tmp_path,monkeypatch);path=target/'baseline_reference.npz'
    config=json.loads((target/'metrics.json').read_text());values=judge.evaluate(path)['metrics']
    # Slightly perturb reference to make all errors and NCC departures nonzero.
    with np.load(path) as z:out={k:z[k].copy() for k in z.files}
    out['reconstruction'][0,0]*=1.00001;out['emissivity'][0,0]*=1.00001
    prediction=tmp_path/'prediction.npz';np.savez(prediction,**out)
    values=judge.evaluate(prediction)['metrics']
    for key in judge.METRICS:
        changed=dict(config);changed[key+'_boundary']=(values[key]+1)/2 if key.endswith('_ncc') else values[key]*.5
        (target/'metrics.json').write_text(json.dumps(changed))
        assert not judge.evaluate(prediction)['passed'],key
    (target/'metrics.json').write_text(json.dumps(config))

def test_hidden_reference_is_not_used(tmp_path,monkeypatch):
    target=sandbox(tmp_path,monkeypatch)
    with np.load(target/'baseline_reference.npz') as z:out={k:z[k]*1.1 for k in z.files}
    np.savez(target/'ground_truth.npz',**out)
    assert judge.evaluate(target/'baseline_reference.npz')['passed']

@pytest.mark.parametrize('value',[-1.01,-1.])
def test_negative_physical_boundary(tmp_path,monkeypatch,value):
    target=sandbox(tmp_path,monkeypatch);config=json.loads((target/'metrics.json').read_text())
    config['density_observation_rms_boundary']=value;(target/'metrics.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):judge.evaluate(target/'baseline_reference.npz')

def test_nonfinite_computed_metric_is_error(tmp_path):
    path=tmp_path/'overflow.npz';np.savez(path,reconstruction=np.full((1,35),1e308),emissivity=np.full((1,35),1e308))
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError):judge.evaluate(path)

def test_cli_foreign_directory_and_errors(tmp_path):
    report=tmp_path/'nested/report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)]
    result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(report.read_text())['passed']
    cmd[3]=str(tmp_path/'missing.npz');result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2
    payload=json.loads(report.read_text());assert payload['status']=='error' and payload['metrics']=={}
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
