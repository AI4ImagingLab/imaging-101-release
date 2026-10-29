from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_ultrasound_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as d:return d['reconstruction'].copy()

def test_known_metrics_and_degenerate_reference():
    r=np.array([1.,2.,3.]);x=np.array([2.,3.,4.]);m=judge.field_metrics(x,r)
    assert m['nrmse']==pytest.approx(.5);assert m['ncc']==pytest.approx(20/np.sqrt(14*29))
    assert judge.field_metrics(np.zeros(3),r)['ncc']==0
    with pytest.raises(ValueError):judge.field_metrics(x,np.ones(3))

@pytest.mark.parametrize('kind',['zero','uniform','sign','swap','reversed','vx_only_wrong','vz_only_wrong','outside'])
def test_poor_complete_outputs_fail(tmp_path,kind):
    a=reference()
    if kind=='zero':a[:]=0
    elif kind=='uniform':a[:]=.1
    elif kind=='sign':a=-a
    elif kind=='swap':a=a[:,::-1].copy()
    elif kind=='reversed':a=a[:,:,:,::-1].copy()
    elif kind=='vx_only_wrong':a[:,0]*=1.1
    elif kind=='vz_only_wrong':a[:,1]*=1.1
    elif kind=='outside':a[:,:,0,0]=.1
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=a)
    report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail' and report['metrics'] and not report['errors']

@pytest.mark.parametrize('kind',['missing','wrongkey','extra','shape','integer','float16','complex','object','nan','inf'])
def test_invalid_contract(tmp_path,kind):
    a=reference();data={'reconstruction':a}
    if kind=='missing':data={}
    elif kind=='wrongkey':data={'velocity':a}
    elif kind=='extra':data['other']=a
    elif kind=='shape':data['reconstruction']=a[0]
    elif kind in ['integer','float16','complex','object']:data['reconstruction']=a.astype({'integer':'int64','float16':'float16','complex':'complex128','object':'object'}[kind])
    elif kind=='nan':a[0,0,4,5]=np.nan
    elif kind=='inf':a[0,1,4,5]=np.inf
    p=tmp_path/'invalid.npz';np.savez(p,**data);report,code=judge.evaluate(p)
    assert code==2 and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('kind',['missing','direction','zero','nan','bad_ncc'])
def test_undefined_metric_configurations(tmp_path,kind):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if kind=='missing':config['thresholds'].pop('vz_ncc')
    elif kind=='direction':config['thresholds']['vx_ncc']['comparison']='<='
    elif kind=='zero':config['thresholds']['vx_nrmse']['boundary']=0
    elif kind=='nan':config['thresholds']['weighted_phasor_rms']['boundary']=float('nan')
    elif kind=='bad_ncc':config['thresholds']['vx_ncc']['boundary']=1.1
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=tmp_path)[1]==2

def test_missing_corrupt_and_evaluator_assets(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')[1]==2
    (tmp_path/'corrupt.npz').write_bytes(b'invalid')
    assert judge.evaluate(tmp_path/'corrupt.npz')[1]==2
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=tmp_path)[1]==2
    shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=tmp_path)[1]==2

def test_foreign_cwd_and_report_write_failure(tmp_path):
    prediction=tmp_path/'field.npz';np.savez(prediction,reconstruction=reference())
    report=tmp_path/'nested'/'report.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True,shell=False)
    assert run.returncode==0 and json.loads(report.read_text())['passed']
    run=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,shell=False)
    assert run.returncode==2 and b'Cannot write report' in run.stderr

def test_float32_submission_and_independent_rf_physics(tmp_path):
    p=tmp_path/'float32.npz';np.savez(p,reconstruction=reference().astype('float32'))
    report,code=judge.evaluate(p);assert code==0
    assert report['metrics']['weighted_phasor_rms']==pytest.approx(.05201278063077122,abs=2e-9)
