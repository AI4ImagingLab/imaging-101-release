from pathlib import Path
import importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def ref():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return z['reconstruction']

def test_independently_known_metric_cases():
    a=np.array([0.,1.,2.]);out=judge.reference_metrics(2*a,a)
    assert np.isclose(out['ncc_vs_ref'],1)
    assert np.isclose(out['nrmse_vs_ref'],np.sqrt(5/3)/2)
    assert judge.reference_metrics(np.zeros(3),a)['ncc_vs_ref']==0

def test_malformed_inputs(tmp_path):
    a=ref();cases=[{}, {'reconstruction':a.astype(np.float32)},{'reconstruction':a.astype(complex)},{'reconstruction':a[0]}]
    for value in [np.nan,np.inf,-.1]:
        bad=a.copy();bad[0,0,0,0]=value;cases.append({'reconstruction':bad})
    for i,case in enumerate(cases):
        path=tmp_path/f'bad{i}.npz';np.savez(path,**case);report=tmp_path/f'report{i}.json'
        assert judge.evaluate(path,report)==2;result=json.loads(report.read_text());assert result['metrics']=={} and result['errors']
    assert judge.evaluate(tmp_path/'absent',tmp_path/'absentreport.json')==2

def test_quality_failure_and_report_write_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'observation_residual',lambda x:1.)
    p=tmp_path/'zero.npz';np.savez(p,reconstruction=np.zeros_like(ref()));report=tmp_path/'nested/report.json'
    assert judge.evaluate(p,report)==1;assert json.loads(report.read_text())['status']=='fail'
    assert judge.evaluate(p,tmp_path)==2

def test_missing_evaluator_and_actual_truth_priority(tmp_path,monkeypatch):
    home=tmp_path/'hidden';home.mkdir();(home/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text());monkeypatch.setattr(judge,'HOME',home);monkeypatch.setattr(judge,'observation_residual',lambda x:.1)
    a=ref();p=tmp_path/'good.npz';np.savez(p,reconstruction=a)
    assert judge.evaluate(p,tmp_path/'missing.json')==2
    np.savez(home/'baseline_reference.npz',reconstruction=a*.9);np.savez(home/'ground_truth.npz',reconstruction=a)
    assert judge.evaluate(p,tmp_path/'truth.json')==0
    np.savez(home/'ground_truth.npz',reconstruction=np.zeros_like(a))
    assert judge.evaluate(p,tmp_path/'degenerate.json')==2
