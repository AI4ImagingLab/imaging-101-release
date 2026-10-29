import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metric_case():
    a=np.array([1.,2.,3.]);v=judge.compare(a*2,a)
    assert v['ncc']==pytest.approx(1);assert v['nrmse']==pytest.approx(np.sqrt(14/3)/2)

def test_good_and_bad(tmp_path):
    ref=np.load(ROOT/'evaluation/baseline_reference.npz');p=tmp_path/'p.npz'
    np.savez(p,reconstruction=ref['reconstruction'],transmission=ref['transmission']);assert judge.evaluate(p)['passed']
    np.savez(p,reconstruction=np.zeros((1,3,35,35)),transmission=np.zeros((1,35,35)));assert judge.evaluate(p)['status']=='fail'
    np.savez(p,reconstruction=np.full((1,3,35,35),1e300),transmission=np.full((1,35,35),1e300));assert judge.evaluate(p)['status']=='fail'

@pytest.mark.parametrize('problem',['missing','shape','dtype','complex','nan','inf'])
def test_malformed(tmp_path,problem):
    ref=np.load(ROOT/'evaluation/baseline_reference.npz');c=ref['reconstruction'].copy();t=ref['transmission'].copy();p=tmp_path/'p.npz'
    if problem=='missing':np.savez(p,reconstruction=c)
    else:
        if problem=='shape':c=c[0]
        if problem=='dtype':c=c.astype(np.int64)
        if problem=='complex':c=c.astype(complex)
        if problem=='nan':c.flat[0]=np.nan
        if problem=='inf':c.flat[0]=np.inf
        np.savez(p,reconstruction=c,transmission=t)
    with pytest.raises(ValueError):judge.evaluate(p)

def test_error_report_and_missing_file(tmp_path):
    import json
    report=tmp_path/'new/report.json';assert judge.main(['--prediction',str(tmp_path/'missing.npz'),'--report',str(report)])==2
    result=json.loads(report.read_text());assert result['status']=='error' and result['metrics']=={} and result['errors']

def test_report_write_failure(tmp_path):
    assert judge.main(['--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path)])==2

def test_missing_evaluator_asset(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    with pytest.raises(FileNotFoundError):judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
