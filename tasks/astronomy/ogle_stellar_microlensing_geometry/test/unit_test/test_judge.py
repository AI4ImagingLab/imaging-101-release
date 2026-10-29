from pathlib import Path
import subprocess,sys,json,shutil,importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return {k:f[k].copy() for k in f.files}
def call(tmp_path,a=None,judge=None):
    p=tmp_path/'prediction.npz';report=tmp_path/'report.json'
    if a is not None:np.savez(p,**a)
    r=subprocess.run([sys.executable,str(judge or ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    return r.returncode,json.loads(report.read_text())
def test_reference_and_known_metrics(tmp_path):
    code,r=call(tmp_path,reference());assert code==0 and r['passed'];assert r['metrics']['t0_error']==0
    assert abs(r['metrics']['reduced_chi2']-427.20382192774014/378)<1e-10
@pytest.mark.parametrize('kind',['missing_file','missing_key','shape','int','complex','nan','inf','physical'])
def test_invalid(tmp_path,kind):
    a=reference()
    if kind=='missing_file':a=None
    elif kind=='missing_key':del a['source_blend_flux']
    elif kind=='shape':a['reconstruction']=a['reconstruction'][0]
    elif kind=='int':a['reconstruction']=a['reconstruction'].astype(int)
    elif kind=='complex':a['reconstruction']=a['reconstruction'].astype(complex)
    elif kind=='nan':a['reconstruction'][0,0]=np.nan
    elif kind=='inf':a['source_blend_flux'][0,0]=np.inf
    else:a['reconstruction'][0,1]=-1
    code,r=call(tmp_path,a);assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['epoch','impact','timescale','flux','flat'])
def test_poor(tmp_path,kind):
    a=reference()
    if kind=='epoch':a['reconstruction'][0,0]+=1
    elif kind=='impact':a['reconstruction'][0,1]*=2
    elif kind=='timescale':a['reconstruction'][0,2]*=2
    elif kind=='flux':a['source_blend_flux']*=1.2
    else:a['reconstruction'][0,1]=100
    code,r=call(tmp_path,a);assert code==1 and not r['passed']
def test_actual_truth_precedence_and_bad_reference(tmp_path):
    ev=tmp_path/'evaluation';shutil.copytree(ROOT/'evaluation',ev);shutil.copytree(ROOT/'data',tmp_path/'data');a=reference()
    bad=reference();bad['reconstruction'][0,0]+=10;np.savez(ev/'baseline_reference.npz',**bad);np.savez(ev/'ground_truth.npz',**a)
    assert call(tmp_path,a,ev/'judge.py')[0]==0
    a['reconstruction'][0,2]=-1;np.savez(ev/'ground_truth.npz',**a)
    code,r=call(tmp_path,reference(),ev/'judge.py');assert code==2 and r['metrics']=={}
def test_missing_reference_and_bad_config(tmp_path):
    ev=tmp_path/'evaluation';shutil.copytree(ROOT/'evaluation',ev);(ev/'baseline_reference.npz').unlink()
    assert call(tmp_path,reference(),ev/'judge.py')[0]==2
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',ev);cfg=json.loads((ev/'metrics.json').read_text());cfg['u0_error_boundary']=-1;(ev/'metrics.json').write_text(json.dumps(cfg))
    assert call(tmp_path,reference(),ev/'judge.py')[0]==2
def test_report_write_failure(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**reference())
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True,text=True)
    assert r.returncode==2 and 'Report write failed' in r.stderr
def test_float32_contract(tmp_path):
    a={k:v.astype('float32') for k,v in reference().items()};assert call(tmp_path,a)[0]==0
