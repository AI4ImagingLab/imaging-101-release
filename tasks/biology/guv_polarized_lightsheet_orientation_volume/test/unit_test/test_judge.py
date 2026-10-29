from pathlib import Path
import importlib.util,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

@pytest.fixture
def small_evaluator(tmp_path,monkeypatch):
    monkeypatch.setattr(J,'HERE',tmp_path);monkeypatch.setattr(J,'SHAPE',(1,2,2,2,15))
    monkeypatch.setattr(J,'known_operator_and_observations',lambda:({},None))
    monkeypatch.setattr(J,'observation_residual',lambda f,h,g:0.)
    r=(np.arange(120,dtype=float).reshape(1,2,2,2,15)+1)/100
    np.savez(tmp_path/'baseline_reference.npz',reconstruction=r)
    cfg={'ncc_boundary':.99,'nrmse_boundary':.1,'max_abs_boundary':.1,'forward_l2_boundary':.3}
    (tmp_path/'metrics.json').write_text(json.dumps(cfg));return tmp_path,r

@pytest.mark.parametrize('case',['missing','key','extra','shape','integer','float32','complex','nan','inf','object'])
def test_malformed_prediction(case,small_evaluator):
    p,r=small_evaluator;pred=p/'prediction.npz'
    if case!='missing':
        if case=='key':np.savez(pred,wrong=r)
        elif case=='extra':np.savez(pred,reconstruction=r,extra=np.ones(1))
        else:
            x=r.copy()
            if case=='shape':x=x[...,0]
            elif case=='integer':x=x.astype(int)
            elif case=='float32':x=x.astype(np.float32)
            elif case=='complex':x=x.astype(complex)
            elif case=='object':x=x.astype(object)
            elif case=='nan':x.flat[0]=np.nan
            elif case=='inf':x.flat[0]=np.inf
            np.savez(pred,reconstruction=x)
    report=J.evaluate(pred)
    assert report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']

def test_valid_poor_zero_and_all_coefficients_scored(small_evaluator):
    p,r=small_evaluator;pred=p/'prediction.npz';np.savez(pred,reconstruction=np.zeros_like(r))
    q=J.evaluate(pred);assert q['status']=='fail' and len(q['metrics'])==46
    x=r.copy();x[...,14]=0;np.savez(pred,reconstruction=x)
    q=J.evaluate(pred);assert q['status']=='fail' and q['metrics']['coefficient_14_ncc']==0

def test_actual_truth_priority_independent_known_case(small_evaluator):
    p,r=small_evaluator;truth=r+1
    np.savez(p/'ground_truth.npz',reconstruction=truth)
    np.savez(p/'prediction.npz',reconstruction=truth)
    q=J.evaluate(p/'prediction.npz');assert q['status']=='pass'
    assert q['metrics']['coefficient_0_nrmse']==0

def test_invalid_truth_does_not_fallback(small_evaluator):
    p,r=small_evaluator
    np.savez(p/'ground_truth.npz',reconstruction=np.zeros((1,)))
    np.savez(p/'prediction.npz',reconstruction=r)
    q=J.evaluate(p/'prediction.npz');assert q['status']=='error' and q['metrics']=={} and 'Evaluator' in q['errors'][0]

def test_invalid_threshold_and_missing_reference(small_evaluator):
    p,r=small_evaluator;np.savez(p/'prediction.npz',reconstruction=r)
    cfg=json.loads((p/'metrics.json').read_text());cfg['ncc_boundary']=None;(p/'metrics.json').write_text(json.dumps(cfg))
    assert J.evaluate(p/'prediction.npz')['status']=='error'

def test_cli_pass_fail_error_and_report_write_failure(small_evaluator):
    p,r=small_evaluator;pred=p/'prediction.npz';report=p/'nested'/'report.json'
    np.savez(pred,reconstruction=r)
    assert J.main(['--prediction',str(pred),'--report',str(report)])==0
    assert json.loads(report.read_text())['passed']
    np.savez(pred,reconstruction=np.zeros_like(r));assert J.main(['--prediction',str(pred),'--report',str(report)])==1
    assert J.main(['--prediction',str(p/'missing.npz'),'--report',str(report)])==2
    occupied=p/'occupied';occupied.write_text('file')
    assert J.main(['--prediction',str(pred),'--report',str(occupied/'report.json')])==2

def test_judge_uses_no_solver_imports():
    code=(ROOT/'evaluation/judge.py').read_text()
    assert 'from src' not in code and 'import src' not in code and 'allow_pickle=False' in code
