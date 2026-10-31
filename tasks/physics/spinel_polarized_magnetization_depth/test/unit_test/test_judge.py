from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_contract',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def base():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:return {k:z[k].copy() for k in z.files}
def save(tmp_path,d):
    p=tmp_path/'prediction.npz';np.savez(p,**d);return p
def test_native_and_float32(tmp_path):
    for dtype in (np.float64,np.float32):
        report,code=judge.evaluate(save(tmp_path,{k:v.astype(dtype) for k,v in base().items()}));assert code==0,report
@pytest.mark.parametrize('bad',['missing_key','shape','integer','complex','nan','inf','physical','negative_profile','object'])
def test_invalid_arrays(tmp_path,bad):
    d=base()
    if bad=='missing_key':del d['parameters']
    elif bad=='shape':d['reconstruction']=d['reconstruction'][0]
    elif bad=='integer':d['parameters']=d['parameters'].astype(int)
    elif bad=='complex':d['parameters']=d['parameters'].astype(complex)
    elif bad=='nan':d['parameters'].flat[0]=np.nan
    elif bad=='inf':d['reconstruction'].flat[0]=np.inf
    elif bad=='physical':d['parameters'][0,5]=31
    elif bad=='negative_profile':d['reconstruction'].flat[0]=-1
    elif bad=='object':d['parameters']=d['parameters'].astype(object)
    report,code=judge.evaluate(save(tmp_path,d));assert code==2 and not report['metrics'] and report['errors']
def test_missing_input_and_poor_physical_output(tmp_path):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    d=base();d['parameters'][0,6]*=1.3
    with np.load(ROOT/'data/raw_data.npz') as z:depth=z['depth'][0]
    d['reconstruction']=judge.depth_profiles(d['parameters'][0],depth)[None]
    report,code=judge.evaluate(save(tmp_path,d));assert code==1 and report['status']=='fail'
def copy_evaluator(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir();shutil.copy(ROOT/'evaluation/metrics.json',folder/'metrics.json');shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz')
    (tmp_path/'data').mkdir();shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');return folder
def test_truth_precedence_and_invalid_reference(tmp_path):
    folder=copy_evaluator(tmp_path);pred=save(tmp_path,base());d=base();d['parameters'][0,6]*=1.05
    with np.load(ROOT/'data/raw_data.npz') as z:depth=z['depth'][0]
    d['reconstruction']=judge.depth_profiles(d['parameters'][0],depth)[None];np.savez(folder/'ground_truth.npz',**d)
    report,code=judge.evaluate(pred,folder);assert code==1 and report['metrics']['parameter_range_error']>.001
    d['parameters'][0,6]=-1;np.savez(folder/'ground_truth.npz',**d);report,code=judge.evaluate(pred,folder);assert code==2 and 'hidden reference' in report['errors'][0]
def test_missing_assets_invalid_config_and_report_write(tmp_path):
    folder=copy_evaluator(tmp_path);pred=save(tmp_path,base());(folder/'baseline_reference.npz').unlink();assert judge.evaluate(pred,folder)[1]==2
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz');cfg=json.loads((folder/'metrics.json').read_text());cfg['parameter_range_error_boundary']=-1;(folder/'metrics.json').write_text(json.dumps(cfg));assert judge.evaluate(pred,folder)[1]==2
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(tmp_path)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==2 and 'Report write failed' in r.stderr
def test_external_cwd_and_json(tmp_path):
    out=tmp_path/'nested/report.json';r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(out)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads(out.read_text())['passed']
