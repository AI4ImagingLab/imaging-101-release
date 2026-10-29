import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('independent_judge',Path('evaluation/judge.py'));j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def save(tmp_path,a,key='reconstruction'):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**{key:a});return p
def reference():
    with np.load('evaluation/baseline_reference.npz',allow_pickle=False) as a:return a['reconstruction']
def test_metric_known_case_and_zero():
    a=np.arange(1.,5.).reshape(1,2,2);b=2*a
    r=j.compare(b,a);assert r['ncc']==pytest.approx(1.);assert r['nrmse']==pytest.approx(np.sqrt(7.5)/3)
    assert j.compare(np.zeros_like(a),a)['ncc']==0
    with pytest.raises(ValueError):j.compare(a,np.ones_like(a))
def test_good_zero_and_scaled_fail(tmp_path):
    r=reference();assert j.evaluate(save(tmp_path,r))[1]==0
    assert j.evaluate(save(tmp_path,np.zeros_like(r)))[1]==1
    assert j.evaluate(save(tmp_path,2*r))[1]==1
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf'])
def test_bad_submissions(tmp_path,kind):
    r=reference()
    if kind=='missing':p=tmp_path/'absent.npz'
    elif kind=='key':p=save(tmp_path,r,'wrong')
    elif kind=='shape':p=save(tmp_path,r[0])
    elif kind=='dtype':p=save(tmp_path,r.astype('float32'))
    elif kind=='complex':p=save(tmp_path,r.astype('complex128'))
    else:r.flat[0]=np.nan if kind=='nan' else np.inf;p=save(tmp_path,r)
    report,code=j.evaluate(p);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
def test_missing_reference_and_bad_config(tmp_path,monkeypatch):
    p=save(tmp_path,reference());monkeypatch.setattr(j,'HERE',tmp_path)
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":0.9,"nrmse_boundary":0.1}')
    assert j.evaluate(p)[1]==2
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":1,"nrmse_boundary":0}')
    assert j.evaluate(p)[1]==2
def test_report_parent_and_write_failure(tmp_path):
    p=save(tmp_path,reference());report=tmp_path/'new'/'report.json'
    assert j.main(['--prediction',str(p),'--report',str(report)])==0 and report.exists()
    assert j.main(['--prediction',str(p),'--report',str(tmp_path)])==2
