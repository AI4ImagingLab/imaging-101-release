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

@pytest.mark.parametrize('bad',['missing_key','shape','integer','complex','nan','inf','negative','padding','gauge','object'])
def test_invalid_predictions(tmp_path,bad):
    d=base();a=d['reconstruction']
    if bad=='missing_key':del d['source_energy']
    elif bad=='shape':d['reconstruction']=a[0]
    elif bad=='integer':d['reconstruction']=a.astype(int)
    elif bad=='complex':d['reconstruction']=a.astype(complex)
    elif bad=='nan':a.flat[0]=np.nan
    elif bad=='inf':a.flat[0]=np.inf
    elif bad=='negative':a.flat[0]=-1
    elif bad=='padding':d['site_amplification'][0,4,0,4]=1
    elif bad=='gauge':d['site_amplification']*=2
    elif bad=='object':d['reconstruction']=a.astype(object)
    report,code=judge.evaluate(save(tmp_path,d));assert code==2 and report['status']=='error' and not report['metrics'] and report['errors']

def test_missing_prediction(tmp_path):
    report,code=judge.evaluate(tmp_path/'absent.npz');assert code==2 and not report['metrics']

def test_physical_poor_prediction(tmp_path):
    d=base();d['reconstruction']*=2
    report,code=judge.evaluate(save(tmp_path,d));assert code==1 and report['status']=='fail'

def evaluator_copy(tmp_path):
    folder=tmp_path/'evaluation';folder.mkdir();shutil.copy(ROOT/'evaluation/metrics.json',folder/'metrics.json');shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz')
    (tmp_path/'data').mkdir();shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');return folder

def test_reference_precedence_and_physicality(tmp_path):
    folder=evaluator_copy(tmp_path);d=base();pred=save(tmp_path,d)
    changed=base();changed['source_energy']*=1.1;np.savez(folder/'ground_truth.npz',**changed)
    report,code=judge.evaluate(pred,folder);assert code==1 and report['metrics']['source_energy_max_abs_log_ratio']>.09
    changed['source_energy']*=-1;np.savez(folder/'ground_truth.npz',**changed)
    report,code=judge.evaluate(pred,folder);assert code==2 and 'hidden reference' in report['errors'][0]

def test_missing_reference_and_bad_config(tmp_path):
    folder=evaluator_copy(tmp_path);pred=save(tmp_path,base());(folder/'baseline_reference.npz').unlink()
    report,code=judge.evaluate(pred,folder);assert code==2
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',folder/'baseline_reference.npz');cfg=json.loads((folder/'metrics.json').read_text());cfg['reconstruction_max_abs_log_ratio_boundary']=0;(folder/'metrics.json').write_text(json.dumps(cfg))
    report,code=judge.evaluate(pred,folder);assert code==2 and not report['metrics']

def test_cli_external_cwd_and_report_failure(tmp_path):
    pred=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads(report.read_text())['passed']
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(tmp_path)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==2 and 'Report write failed' in r.stderr
