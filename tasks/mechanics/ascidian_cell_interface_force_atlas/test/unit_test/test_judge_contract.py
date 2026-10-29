from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_force_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}

def test_known_metric_definitions():
    r=np.array([1.,2.,3.]);x=np.array([2.,3.,4.])
    m=judge.reference_metrics(x,r)
    assert m['nrmse']==pytest.approx(.5)
    assert m['ncc']==pytest.approx(20/np.sqrt(14*29))
    assert judge.reference_metrics(np.zeros(3),r)['ncc']==0
    with pytest.raises(ValueError):judge.reference_metrics(x,np.ones(3))

@pytest.mark.parametrize('kind',['zero','uniform','reversed','wrong_tensions','wrong_pressures'])
def test_bad_outputs_fail_all_quantities(tmp_path,kind):
    data=reference()
    for key,a in data.items():
        if kind=='zero':data[key]=np.zeros_like(a)
        elif kind=='uniform':data[key]=np.full_like(a,a.mean())
        elif kind=='reversed':data[key]=a[:,::-1].copy()
        elif kind=='wrong_tensions' and key=='tensions':data[key]=a*1.2
        elif kind=='wrong_pressures' and key=='pressures':data[key]=a*1.2
    path=tmp_path/'bad.npz';np.savez(path,**data)
    report,code=judge.evaluate(path)
    assert code==1 and report['status']=='fail' and not report['passed']
    assert report['metrics'] and not report['errors']

@pytest.mark.parametrize('kind',['missing_key','wrong_key','wrong_shape','integer','float16','complex','nan','inf','object','extra_key'])
def test_malformed_output_contract(tmp_path,kind):
    data=reference()
    if kind=='missing_key':data.pop('pressures')
    elif kind=='wrong_key':data['reconstruction']=data.pop('tensions')
    elif kind=='wrong_shape':data['tensions']=data['tensions'][0]
    elif kind in ['integer','float16','complex','object']:
        dtype={'integer':'int64','float16':'float16','complex':'complex128','object':'object'}[kind]
        data['tensions']=data['tensions'].astype(dtype)
    elif kind in ['nan','inf']:data['pressures'][0,2]=float(kind)
    elif kind=='extra_key':data['extra']=np.zeros(1)
    path=tmp_path/'invalid.npz';np.savez(path,**data)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_missing_corrupt_and_reference_errors(tmp_path):
    report,code=judge.evaluate(tmp_path/'missing.npz');assert code==2 and report['metrics']=={}
    bad=tmp_path/'corrupt.npz';bad.write_bytes(b'not an npz')
    assert judge.evaluate(bad)[1]==2
    assets=tmp_path/'assets';assets.mkdir()
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=assets)[1]==2
    shutil.copy2(ROOT/'evaluation/metrics.json',assets/'metrics.json')
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=assets)[1]==2
    data=reference();data['pressures'][:]=1
    np.savez(assets/'baseline_reference.npz',**data)
    assert judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=assets)[1]==2

def test_foreign_cwd_cli_and_report_write_failure(tmp_path):
    path=tmp_path/'prediction.npz';np.savez(path,**reference())
    report=tmp_path/'nested'/'report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)]
    run=subprocess.run(cmd,cwd=tmp_path,shell=False,capture_output=True)
    assert run.returncode==0 and json.loads(report.read_text())['passed']
    run=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,shell=False,capture_output=True)
    assert run.returncode==2 and b'Cannot write report' in run.stderr

def test_float32_precision_supported(tmp_path):
    path=tmp_path/'float32.npz';np.savez(path,**{k:v.astype('float32') for k,v in reference().items()})
    assert judge.evaluate(path)[1]==0

@pytest.mark.parametrize('kind',['reversed_direction','bad_ncc','zero_tolerance','nonfinite','missing_metric'])
def test_undefined_metric_configuration_rejected(tmp_path,kind):
    definition=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if kind=='reversed_direction':definition['thresholds']['tensions_ncc']['comparison']='<='
    elif kind=='bad_ncc':definition['thresholds']['tensions_ncc']['boundary']=1.5
    elif kind=='zero_tolerance':definition['thresholds']['tensions_nrmse']['boundary']=0
    elif kind=='nonfinite':definition['thresholds']['pressures_nrmse']['boundary']=float('nan')
    elif kind=='missing_metric':definition['thresholds'].pop('pressures_ncc')
    (tmp_path/'metrics.json').write_text(json.dumps(definition))
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',asset_dir=tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']
