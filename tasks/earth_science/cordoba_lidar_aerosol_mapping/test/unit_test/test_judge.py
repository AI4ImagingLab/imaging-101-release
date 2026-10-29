import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z.files}

def sandbox(tmp_path,monkeypatch):
    e=tmp_path/'evaluation';e.mkdir()
    for name in ['metrics.json','baseline_reference.npz']:(e/name).write_bytes((ROOT/'evaluation'/name).read_bytes())
    (tmp_path/'data').mkdir();(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes());monkeypatch.setattr(j,'HERE',e);return e

def test_known_metrics_and_independent_beer_lambert_case():
    assert_allclose(j.compare(np.array([0.,2.]),np.array([0.,1.])),[1.,np.sqrt(.5)])
    beta=np.zeros((2,20));bm=np.full(20,2e-6);am=np.full(20,1e-5)
    signal=np.tile(np.exp(-2*1e-5*(np.arange(20)-19)*7.5),(2,1))
    assert_allclose(j.return_rms(beta,signal,bm,am),0,atol=1e-15)
    assert_allclose(j.return_rms(beta,signal*23,bm,am),0,atol=1e-15)

def test_both_quantities_consumed_and_valid_poor_rejected(tmp_path):
    a=baseline();p=tmp_path/'p.npz';np.savez(p,**a);assert j.evaluate(p)['passed']
    for key in a:
        changed=baseline();changed[key]=np.zeros_like(changed[key]);np.savez(p,**changed)
        report=j.evaluate(p);assert report['status']=='fail' and not report['passed']

@pytest.mark.parametrize('mutation',['missing','shape','dtype','complex','nan','inf'])
def test_malformed_arrays(tmp_path,mutation):
    a=baseline()
    if mutation=='missing':a.pop('extinction')
    elif mutation=='shape':a['reconstruction']=a['reconstruction'][0]
    elif mutation=='dtype':a['reconstruction']=a['reconstruction'].astype('float32')
    elif mutation=='complex':a['reconstruction']=a['reconstruction'].astype('complex128')
    else:a['reconstruction'][0,0,0]=np.nan if mutation=='nan' else np.inf
    p=tmp_path/'p.npz';np.savez(p,**a)
    with pytest.raises((ValueError,KeyError)):j.evaluate(p)

@pytest.mark.parametrize('value',[None,-1.,float('nan'),float('inf'),True,'1'])
def test_strict_configuration(tmp_path,monkeypatch,value):
    e=sandbox(tmp_path,monkeypatch);c=json.loads((e/'metrics.json').read_text())
    if value is None:c.pop('backscatter_nrmse_boundary')
    else:c['backscatter_nrmse_boundary']=value
    (e/'metrics.json').write_text(json.dumps(c))
    with pytest.raises((ValueError,KeyError)):j.evaluate(e/'baseline_reference.npz')

@pytest.mark.parametrize('boundary',[-1.01,1.01])
def test_cosine_configuration_range(tmp_path,monkeypatch,boundary):
    e=sandbox(tmp_path,monkeypatch);c=json.loads((e/'metrics.json').read_text());c['backscatter_ncc_boundary']=boundary;(e/'metrics.json').write_text(json.dumps(c))
    with pytest.raises(ValueError):j.evaluate(e/'baseline_reference.npz')

def test_actual_boundaries_and_truth_precedence(tmp_path,monkeypatch):
    e=sandbox(tmp_path,monkeypatch);a=baseline();a['reconstruction'][0,0,0]+=3e-11;a['extinction'][0,0,0]+=1.5e-9
    p=tmp_path/'p.npz';np.savez(p,**a);values=j.evaluate(p)['metrics'];config=json.loads((e/'metrics.json').read_text())
    for key in j.METRICS:
        changed=dict(config);changed[key+'_boundary']=(1+values[key])/2 if key.endswith('_ncc') else values[key]*.5
        (e/'metrics.json').write_text(json.dumps(changed));assert not j.evaluate(p)['passed'],key
        np.savez(p,**a)
    (e/'metrics.json').write_text(json.dumps(config));a=baseline();a['reconstruction']*=1.1;np.savez(e/'ground_truth.npz',**a)
    assert not j.evaluate(e/'baseline_reference.npz')['passed']

def test_nonfinite_arithmetic_and_foreign_cli(tmp_path):
    a=baseline();a['reconstruction'].fill(1e308);p=tmp_path/'p.npz';np.savez(p,**a)
    with np.errstate(all='ignore'):
        with pytest.raises(ValueError):j.evaluate(p)
    report=tmp_path/'nested/report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    cmd[3]=str(tmp_path/'missing.npz');assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
    assert json.loads(report.read_text())['metrics']=={}
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
