from pathlib import Path
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
@pytest.mark.parametrize('kind',['missing','key','shape','integer','complex','float16','object','nan','inf','negative'])
def test_invalid_contract(tmp_path,kind):
    p=tmp_path/'p.npz';a=np.zeros((1,171))
    if kind=='missing':pass
    elif kind=='key':np.savez(p,other=a)
    else:
        if kind=='shape':a=a[:,1:]
        elif kind=='integer':a=a.astype(int)
        elif kind=='complex':a=a.astype(complex)
        elif kind=='float16':a=a.astype(np.float16)
        elif kind=='object':a=a.astype(object)
        elif kind in ['nan','inf']:a[0,0]=float(kind)
        elif kind=='negative':a[0,0]=-1
        np.savez(p,reconstruction=a)
    r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
def test_zero_law_fails(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,reconstruction=np.zeros((1,171)));r,c=j.evaluate(p)
    assert c==1 and not r['passed']
def test_independent_reference_passes(tmp_path):
    p=tmp_path/'p.npz';a=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];np.savez(p,reconstruction=a)
    r,c=j.evaluate(p);assert c==0 and r['passed']

def test_float32_accurate_reference_passes(tmp_path):
    p=tmp_path/'p.npz'
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a=z['reconstruction'].astype(np.float32)
    np.savez(p,reconstruction=a);r,c=j.evaluate(p);assert c==0 and r['passed']

def test_valid_huge_finite_coefficients_fail_with_finite_metrics(tmp_path):
    import json,warnings
    p=tmp_path/'p.npz';np.savez(p,reconstruction=np.full((1,171),1e308))
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        r,c=j.evaluate(p)
    assert c==1 and r['status']=='fail' and not r['errors']
    assert all(np.isfinite(v) for v in r['metrics'].values())
    assert r['metrics']['energy_relative_l2']>1e100
    json.dumps(r,allow_nan=False)

def test_stable_metric_helpers_on_known_scalar_cases():
    assert j.relative_l2(np.eye(3),np.array([3.,4.,0.]),np.array([0.,4.,0.]))==pytest.approx(.75,rel=1e-14)
    assert j.relative_l2(np.eye(3),np.full(3,1e308),np.ones(3))==pytest.approx(1e308,rel=1e-12)
    assert j.relative_l2(np.eye(3)*1e308,np.full(3,1e308),np.ones(3))==j.MAX_FLOAT
    assert j.weak_residual_relative_l2(np.array([2.]),np.array([[2.]]),np.array([1.]),2.)==pytest.approx(np.sqrt(2.))
    assert j.weak_residual_relative_l2(np.array([1e308]),np.array([[2.]]),np.array([1.]),2.)==pytest.approx(1e308/np.sqrt(2),rel=1e-12)

def baseline_arrays():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

@pytest.mark.parametrize('kind',['missing_Q','Q_shape','G_shape','H_shape','b_dtype','C_shape','C_nonpositive','Q_nonfinite','H_nonfinite','energy_shape','energy_dtype','energy_zero','stress_zero'])
def test_hidden_asset_validation_before_scoring(tmp_path,monkeypatch,kind):
    import shutil
    arrays=baseline_arrays();prediction=tmp_path/'p.npz';np.savez(prediction,reconstruction=arrays['reconstruction'])
    if kind=='missing_Q':del arrays['Q']
    elif kind=='Q_shape':arrays['Q']=arrays['Q'][:-1]
    elif kind=='G_shape':arrays['G']=arrays['G'][:,:,:3]
    elif kind=='H_shape':arrays['H']=arrays['H'][:-1]
    elif kind=='b_dtype':arrays['b']=arrays['b'].astype(np.float32)
    elif kind=='C_shape':arrays['C']=arrays['C'][None]
    elif kind=='C_nonpositive':arrays['C']=np.array(0.,dtype=np.float64)
    elif kind=='Q_nonfinite':arrays['Q'][0,0]=np.inf
    elif kind=='H_nonfinite':arrays['H'][0,0]=np.nan
    elif kind=='energy_shape':arrays['energy']=arrays['energy'][0]
    elif kind=='energy_dtype':arrays['energy']=arrays['energy'].astype(np.float32)
    elif kind=='energy_zero':arrays['energy']*=0
    elif kind=='stress_zero':arrays['stress']*=0
    np.savez(tmp_path/'baseline_reference.npz',**arrays);shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    monkeypatch.setattr(j,'ROOT',tmp_path);r,c=j.evaluate(prediction)
    assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_actual_truth_takes_precedence_and_malformed_truth_does_not_fallback(tmp_path,monkeypatch):
    import shutil
    arrays=baseline_arrays();p=tmp_path/'p.npz';np.savez(p,reconstruction=arrays['reconstruction'])
    np.savez(tmp_path/'baseline_reference.npz',**arrays);shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    # Temporary synthetic evaluator case, not a released actual-truth claim.
    np.savez(tmp_path/'ground_truth.npz',energy=2*arrays['energy'],stress=2*arrays['stress'])
    monkeypatch.setattr(j,'ROOT',tmp_path);r,c=j.evaluate(p)
    assert c==1 and r['metrics']['energy_relative_l2']==pytest.approx(.5,abs=1e-12)
    np.savez(tmp_path/'ground_truth.npz',energy=arrays['energy'])
    r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

def test_actual_truth_can_carry_operators_without_baseline(tmp_path,monkeypatch):
    import shutil
    arrays=baseline_arrays();p=tmp_path/'p.npz';np.savez(p,reconstruction=arrays['reconstruction'])
    np.savez(tmp_path/'ground_truth.npz',**arrays);shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    monkeypatch.setattr(j,'ROOT',tmp_path);r,c=j.evaluate(p);assert c==0 and r['passed']
def test_unavailable_evaluator_assets(tmp_path,monkeypatch):
    p=tmp_path/'p.npz';np.savez(p,reconstruction=np.zeros((1,171)));monkeypatch.setattr(j,'ROOT',tmp_path)
    r,c=j.evaluate(p);assert c==2 and r['errors']
def test_cli_paths_and_report_failure(tmp_path):
    import subprocess,sys,json
    p=tmp_path/'p.npz';np.savez(p,reconstruction=np.zeros((1,171)))
    report=tmp_path/'new/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert run.returncode==1 and json.loads(report.read_text())['status']=='fail'
    blocker=tmp_path/'blocker';blocker.write_text('file')
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(blocker/'report.json')],capture_output=True)
    assert run.returncode==2 and b'Report write failed' in run.stderr
