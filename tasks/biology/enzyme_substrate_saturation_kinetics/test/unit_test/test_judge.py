from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def sample():return j.load_arrays(ROOT/'evaluation/baseline_reference.npz')
@pytest.mark.parametrize('kind',['missing_key','shape','float32','complex','nan','inf'])
def test_invalid_submission(kind,tmp_path):
    d=sample()
    if kind=='missing_key':del d['maximum_rate']
    elif kind=='shape':d['reconstruction']=d['reconstruction'].ravel()
    elif kind=='float32':d['reconstruction']=d['reconstruction'].astype('float32')
    elif kind=='complex':d['reconstruction']=d['reconstruction'].astype('complex128')
    elif kind=='nan':d['reconstruction'][0,0]=np.nan
    elif kind=='inf':d['reconstruction'][0,0]=np.inf
    p=tmp_path/'x.npz';np.savez(p,**d);r,c=j.evaluate(p)
    assert c==2 and r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']
def test_missing_input(tmp_path):assert j.evaluate(tmp_path/'none.npz')[1]==2
def test_bad_and_good_results(tmp_path):
    d=sample();p=tmp_path/'x.npz';np.savez(p,**d);assert j.evaluate(p)[1]==0
    d['reconstruction']*=2;np.savez(p,**d);r,c=j.evaluate(p);assert c==1 and not r['passed']
def test_metrics_known_case():
    d=sample();s=np.array([[.05,.1,.25,.5,1,2.5,5,8,20,30]])
    p={k:v.copy() for k,v in d.items()};p['maximum_rate']*=1.1;p['fitted_rate']*=1.1
    m=j.metrics(p,d,s);assert abs(m['Vmax_max_relative_error']-.1)<1e-14
    assert m['output_forward_consistency_max_absolute_error']<1e-10
def test_missing_assets_and_invalid_thresholds(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'HERE',tmp_path)
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
    (tmp_path/'metrics.json').write_text(json.dumps({'thresholds':{'wrong':{'comparison':'<=','boundary':1}}}))
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
def test_report_write_failure(tmp_path):
    assert j.main(['--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path)])==2
@pytest.mark.parametrize('kind',['ground_truth_precedence','negative_reference','inconsistent_reference'])
def test_reference_contract(kind,tmp_path,monkeypatch):
    import shutil
    hidden=tmp_path/'evaluation';hidden.mkdir();(tmp_path/'data').mkdir()
    shutil.copy2(ROOT/'evaluation/metrics.json',hidden/'metrics.json')
    shutil.copy2(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    d=sample();np.savez(hidden/'baseline_reference.npz',**d)
    if kind=='ground_truth_precedence':
        del d['maximum_rate'];np.savez(hidden/'ground_truth.npz',**d)
    elif kind=='negative_reference':
        d['reconstruction'][0,0]=-1;np.savez(hidden/'baseline_reference.npz',**d)
    else:
        d['fitted_rate']*=2;np.savez(hidden/'baseline_reference.npz',**d)
    monkeypatch.setattr(j,'HERE',hidden)
    report,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==2 and report['metrics']=={} and report['errors']
