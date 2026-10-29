import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest

TASK=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',TASK/'evaluation/judge.py')
J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

@pytest.fixture
def known_assets(tmp_path,monkeypatch):
    root=tmp_path/'evaluation';root.mkdir()
    truth=np.tile(np.arange(71*72,dtype=float).reshape(1,1,71,72),(1,25,1,1))
    np.savez(root/'baseline_reference.npz',reconstruction=truth+10)
    np.savez(root/'ground_truth.npz',reconstruction=truth)
    (root/'metrics.json').write_text(json.dumps(dict(ncc_boundary=.999,nrmse_boundary=.001,forward_weighted_rms_boundary=1.)))
    pred=tmp_path/'prediction.npz';np.savez(pred,reconstruction=truth)
    # Isolate reference/metric contract from the expensive, separately tested physics check.
    monkeypatch.setattr(J,'forward_rms',lambda *args:0.)
    return root,pred,truth

def test_actual_truth_priority_and_independent_known_metrics(known_assets):
    root,pred,truth=known_assets
    report,code=J.evaluate(pred,root)
    assert code==0 and report['metrics']['nrmse_max']==0
    np.savez(pred,reconstruction=truth+1)
    report,code=J.evaluate(pred,root)
    assert report['metrics']['nrmse_max']==pytest.approx(1/5111)

def test_judge_and_notebook_metric_helper_share_known_case(known_assets):
    from src.visualization import map_metrics
    root,pred,_=known_assets
    reference=np.tile(np.array([1.,0.]),np.prod(J.SHAPE)//2).reshape(J.SHAPE)
    estimate=np.ones(J.SHAPE,dtype=float)
    np.savez(root/'ground_truth.npz',reconstruction=reference)
    np.savez(pred,reconstruction=estimate)
    report,code=J.evaluate(pred,root)
    ncc,error=map_metrics(estimate,reference)
    assert code==1
    assert report['metrics']['ncc_min']==pytest.approx(1/np.sqrt(2))
    assert report['metrics']['nrmse_max']==pytest.approx(1/np.sqrt(2))
    np.testing.assert_allclose(ncc,1/np.sqrt(2),rtol=1e-14)
    np.testing.assert_allclose(error,1/np.sqrt(2),rtol=1e-14)

def test_invalid_actual_truth_does_not_fallback(known_assets):
    root,pred,truth=known_assets
    np.savez(root/'ground_truth.npz',wrong=truth)
    report,code=J.evaluate(pred,root)
    assert code==2 and report['metrics']=={} and 'Reference' in report['errors'][0]

def test_extreme_finite_values_report_arithmetic_error(known_assets):
    root,pred,x=known_assets
    np.savez(pred,reconstruction=np.full_like(x,1e308))
    with np.errstate(over='ignore',invalid='ignore'):
        report,code=J.evaluate(pred,root)
    assert code==2 and report['metrics']=={} and not report['passed']
    assert 'Nonfinite metric arithmetic' in report['errors'][0]

@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','integer','nan','inf','object'])
def test_malformed_predictions(known_assets,kind):
    root,pred,x=known_assets
    if kind=='missing':pred.unlink()
    elif kind=='key':np.savez(pred,wrong=x)
    elif kind=='shape':np.savez(pred,reconstruction=x[...,0])
    else:
        x=x.copy()
        if kind=='float32':x=x.astype(np.float32)
        elif kind=='complex':x=x.astype(complex)
        elif kind=='integer':x=x.astype(int)
        elif kind=='object':x=x.astype(object)
        elif kind=='nan':x[0,0,0,0]=np.nan
        elif kind=='inf':x[0,0,0,0]=np.inf
        np.savez(pred,reconstruction=x)
    report,code=J.evaluate(pred,root)
    assert code==2 and not report['passed'] and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('key,value',[
    ('ncc_boundary',True),('ncc_boundary','0.9'),('ncc_boundary',1.01),('ncc_boundary',-1.01),
    ('ncc_boundary',float('nan')),('nrmse_boundary',False),('nrmse_boundary',-1),
    ('nrmse_boundary',float('inf')),('forward_weighted_rms_boundary',True),
    ('forward_weighted_rms_boundary','1'),('forward_weighted_rms_boundary',-1),
    ('forward_weighted_rms_boundary',None)])
def test_bad_configuration(known_assets,key,value):
    root,pred,_=known_assets;p=root/'metrics.json';config=json.loads(p.read_text());config[key]=value;p.write_text(json.dumps(config))
    report,code=J.evaluate(pred,root)
    assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('kind',['zero','uniform','shuffle','sign','time_reverse'])
def test_poor_fields_fail_full_field_gate(known_assets,kind):
    root,pred,x=known_assets
    if kind=='zero':x=np.zeros_like(x)
    elif kind=='uniform':x=np.full_like(x,x.mean())
    elif kind=='shuffle':x=x[:,:,:,::-1].copy()
    elif kind=='sign':x=-x
    else:
        # Each hour must use its own reference; make a nontrivial time sequence.
        x=x+np.arange(25)[None,:,None,None]*100
        np.savez(root/'ground_truth.npz',reconstruction=x)
        x=x[:,::-1].copy()
    np.savez(pred,reconstruction=x);report,code=J.evaluate(pred,root)
    assert code==1 and report['status']=='fail'

def test_evaluator_missing_reference_and_degenerate(known_assets):
    root,pred,x=known_assets
    (root/'ground_truth.npz').unlink();(root/'baseline_reference.npz').unlink()
    assert J.evaluate(pred,root)[1]==2
    np.savez(root/'ground_truth.npz',reconstruction=np.ones_like(x))
    assert J.evaluate(pred,root)[1]==2

def test_foreign_cwd_and_report_write_failure(tmp_path):
    pred=tmp_path/'zero.npz';np.savez(pred,reconstruction=np.zeros(J.SHAPE,dtype=np.float64))
    report=tmp_path/'nested/report.json';cmd=[sys.executable,str(TASK/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)]
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True)
    assert r.returncode==1 and json.loads(report.read_text())['status']=='fail'
    bad=tmp_path/'a_file';bad.write_text('not a directory')
    r=subprocess.run(cmd[:-1]+[str(bad/'report.json')],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and b'Report write failed' in r.stderr
