from pathlib import Path
import json,importlib.util,subprocess,sys
import numpy as np
import pytest
import shutil
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_known_cosine_and_dynamic_range_nrmse():
    n,r=j.compare(np.array([1.,2.,3.]),np.array([3.,2.,1.]))
    assert n==pytest.approx(10/14);assert r==pytest.approx(np.sqrt(8/3)/2)
    assert j.compare(np.zeros(3),np.array([1.,2.,3.]))[0]==0
    with pytest.raises(ValueError):j.compare(np.ones(3),np.ones(3))

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','object'])
def test_malformed_prediction_is_error_without_scores(tmp_path,kind):
    p=tmp_path/'bad.npz'
    if kind=='missing':pass
    elif kind=='key':np.savez(p,other=np.ones(2))
    elif kind=='object':np.savez(p,reconstruction=np.array([object()]))
    else:
        a=np.zeros(j.SHAPE,dtype=np.float32) if kind in ('nan','inf') else np.zeros((1,2),dtype=np.float32)
        if kind=='dtype':a=a.astype(np.float64)
        if kind=='complex':a=a.astype(np.complex64)
        if kind=='nan':a.flat[0]=np.nan
        if kind=='inf':a.flat[0]=np.inf
        np.savez_compressed(p,reconstruction=a)
    report,code=j.evaluate(p)
    assert code==2 and report['status']=='error' and not report['passed'] and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('bad',[None,True,'0.1',float('nan'),float('inf'),-1.])
def test_illegal_error_boundaries_are_evaluator_errors(tmp_path,monkeypatch,bad):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['nrmse_boundary']=bad
    (tmp_path/'metrics.json').write_text(json.dumps(config));monkeypatch.setattr(j,'ASSETS',tmp_path)
    report,code=j.evaluate('unused.npz');assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('bad',[-1.01,1.01])
def test_cosine_boundary_range(tmp_path,monkeypatch,bad):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['ncc_boundary']=bad
    (tmp_path/'metrics.json').write_text(json.dumps(config));monkeypatch.setattr(j,'ASSETS',tmp_path)
    assert j.evaluate('unused.npz')[1]==2

def test_valid_reference_and_observation_phase_closure():
    report,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==0 and len(report['metrics'])==10
    assert report['metrics']['phase_phasor_rms']<2e-6
    assert report['metrics']['outside_mask_max']==0

def test_zero_valid_submission_fails_and_report_write_error(tmp_path):
    p=tmp_path/'zero.npz';np.savez_compressed(p,reconstruction=np.zeros(j.SHAPE,np.float32))
    report,code=j.evaluate(p);assert code==1 and report['status']=='fail'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','missing.npz','--report',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==2 and result.stderr

def test_actual_truth_priority_missing_and_malformed_reference(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text());monkeypatch.setattr(j,'ASSETS',tmp_path)
    report,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert code==2
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'ground_truth.npz')
    np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=np.ones((1,2)))
    # Isolate asset precedence from the separately tested full physical scoring.
    monkeypatch.setattr(j,'observation_metrics',lambda field:dict(phase_phasor_rms=0.,outside_mask_max=0.))
    report,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert code==0
    np.savez_compressed(tmp_path/'ground_truth.npz',reconstruction=np.ones((1,2)))
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    report,code=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert code==2 and report['metrics']=={}
    assert 'ground_truth' in report['errors'][0] or 'shape' in report['errors'][0]

def test_public_acquisition_constants_and_batch_contract():
    from src.preprocessing import PUBLIC_SHAPE,VENC_M_S,VOXEL_MM
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert str(PUBLIC_SHAPE).replace(' ','') in text.replace(' ','')
    assert str(VENC_M_S) in text
    np.testing.assert_allclose(VOXEL_MM,[1.5,1.492537313432836,1.4814814814814814],atol=1e-15)
    for parameter in ['0.0047','0.007','13 by10','1000','150,100,80']:assert parameter in text

def test_foreign_cwd_cli_and_parent_creation(tmp_path):
    report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    d=json.loads(report.read_text());assert d['passed'] and d['status']=='pass'

def test_every_threshold_direction_and_boundary_is_consumed():
    names=j.METRIC_NAMES;metrics={n:.5 for n in names};limits={n:{'comparison':'>=' if n.endswith('ncc') else '<=','boundary':.5} for n in names}
    assert j.passes(metrics,limits)
    for n in names:
        values=metrics.copy();values[n]=.4 if n.endswith('ncc') else .6
        assert not j.passes(values,limits),n
