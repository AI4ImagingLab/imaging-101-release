import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('standalone_judge',Path('evaluation/judge.py'));j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def reference():
    with np.load('evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def save(tmp_path,data):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**data);return p
def test_metrics_known_vectors():
    a=np.arange(1.,5.);n,e=j.compare(2*a,a);assert n==pytest.approx(1.);assert e==pytest.approx(np.sqrt(7.5)/3)
    assert j.compare(np.zeros_like(a),a)[0]==0
    with pytest.raises(ValueError):j.compare(a,np.ones_like(a))
def test_all_components_are_evaluated(tmp_path):
    r=reference();assert j.evaluate(save(tmp_path,r))[1]==0
    for k in j.KEYS:
        d=reference();d[k]*=0;report,code=j.evaluate(save(tmp_path,d));assert code==1 and len(report['metrics'])==6
        d=reference();d[k]*=2;assert j.evaluate(save(tmp_path,d))[1]==1
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf'])
def test_bad_inputs(tmp_path,kind):
    d=reference()
    if kind=='missing':p=tmp_path/'absent.npz'
    else:
        if kind=='key':del d['dust']
        elif kind=='shape':d['dust']=d['dust'][0]
        elif kind=='dtype':d['dust']=d['dust'].astype('float32')
        elif kind=='complex':d['dust']=d['dust'].astype('complex128')
        else:d['dust'].flat[0]=np.nan if kind=='nan' else np.inf
        p=save(tmp_path,d)
    report,code=j.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']
def test_evaluator_failure_and_report_paths(tmp_path,monkeypatch):
    import json
    p=save(tmp_path,reference());report=tmp_path/'a'/'report.json'
    assert j.main(['--prediction',str(p),'--report',str(report)])==0 and report.exists()
    assert j.main(['--prediction',str(p),'--report',str(tmp_path)])==2
    c=json.loads(Path('evaluation/metrics.json').read_text());(tmp_path/'metrics.json').write_text(json.dumps(c));monkeypatch.setattr(j,'HERE',tmp_path)
    assert j.evaluate(p)[1]==2
    c['dust_nrmse_boundary']=0;(tmp_path/'metrics.json').write_text(json.dumps(c));assert j.evaluate(p)[1]==2
