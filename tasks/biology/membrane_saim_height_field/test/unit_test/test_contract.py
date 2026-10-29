from pathlib import Path
import json,subprocess,sys,shutil,os
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}
def judge(prediction,report,cwd):
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=cwd,capture_output=True,text=True,encoding='utf8')
    return run,json.loads(report.read_text(encoding='utf8'))
def test_valid_prediction_foreign_cwd(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,**reference());run,r=judge(p,tmp_path/'nested/report.json',tmp_path)
    assert run.returncode==0 and r['passed'] and len(r['metrics'])==7
@pytest.mark.parametrize('kind',['missing_file','missing_key','extra_key','shape','dtype','complex','nan','inf','huge_finite','negative'])
def test_malformed_submission(tmp_path,kind):
    p=tmp_path/'prediction.npz';out=reference()
    if kind=='missing_key':out.pop('amplitude')
    elif kind=='extra_key':out['truth']=np.ones(1)
    elif kind=='shape':out['reconstruction']=out['reconstruction'][0]
    elif kind=='dtype':out['reconstruction']=out['reconstruction'].astype(np.float32)
    elif kind=='complex':out['reconstruction']=out['reconstruction'].astype(complex)
    elif kind in ['nan','inf']:out['reconstruction'][0,0,0]=np.nan if kind=='nan' else np.inf
    elif kind=='huge_finite':out['reconstruction'][0,0,0]=1e300
    elif kind=='negative':out['reconstruction'][0,0,0]=-1
    if kind!='missing_file':np.savez(p,**out)
    run,r=judge(p,tmp_path/'report.json',tmp_path)
    assert run.returncode==2 and r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('key',['reconstruction','amplitude','background'])
def test_all_quantities_are_evaluated(tmp_path,key):
    out=reference();out[key]=np.zeros_like(out[key]);p=tmp_path/'prediction.npz';np.savez(p,**out)
    run,r=judge(p,tmp_path/'report.json',tmp_path)
    assert run.returncode==1 and r['status']=='fail' and not r['passed']
def test_missing_hidden_reference_is_evaluator_error(tmp_path):
    e=tmp_path/'evaluation';e.mkdir();shutil.copy2(ROOT/'evaluation/judge.py',e/'judge.py');shutil.copy2(ROOT/'evaluation/metrics.json',e/'metrics.json')
    p=tmp_path/'prediction.npz';np.savez(p,**reference())
    run=subprocess.run([sys.executable,str(e/'judge.py'),'--prediction',str(p),'--report',str(tmp_path/'report.json')],capture_output=True)
    r=json.loads((tmp_path/'report.json').read_text());assert run.returncode==2 and r['metrics']=={} and r['errors']
def test_report_write_failure_returns_two(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,**reference());target=tmp_path/'directory';target.mkdir()
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(target)],capture_output=True)
    assert run.returncode==2 and b'Report write failed' in run.stderr
def test_importing_main_does_not_change_matplotlib_backend():
    cmd='import matplotlib; matplotlib.use("svg"); import src.main; assert matplotlib.get_backend().lower()=="svg"'
    run=subprocess.run([sys.executable,'-c',cmd],cwd=ROOT,capture_output=True);assert run.returncode==0,run.stderr
def test_visible_inputs_do_not_leak_targets_or_github():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:assert set(f.files)=={'angles','intensities'}
    for file in ['README.md','requirements.txt']:
        text=(ROOT/file).read_text(encoding='utf8').lower()
        assert not any(s in text for s in ['github.com','github.io','githubusercontent','codeload.github','git.io'])
