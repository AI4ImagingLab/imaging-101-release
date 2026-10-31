from pathlib import Path
import importlib.util,shutil,subprocess,sys,json
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def reference(): return np.load(T/'evaluation/baseline_reference.npz')['reconstruction']

def test_reference_and_zero_prediction(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,reconstruction=reference());r,c=judge.evaluate(p);assert c==0 and r['passed']
    np.savez(p,reconstruction=np.zeros((1,1228,3)));r,c=judge.evaluate(p);assert c==1 and r['status']=='fail'

@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','not_npz','anchor','angle'])
def test_malformed(tmp_path,case):
    p=tmp_path/'x.npz';x=reference()
    if case=='missing': pass
    elif case=='not_npz': p.write_text('not npz')
    elif case=='key': np.savez(p,wrong=x)
    else:
        if case=='shape': x=x[:,:,:2]
        if case=='dtype': x=x.astype(np.float32)
        if case=='complex': x=x.astype(np.complex128)
        if case=='nan': x[0,1,0]=np.nan
        if case=='inf': x[0,1,0]=np.inf
        if case=='anchor': x[0,0,0]=1
        if case=='angle': x[0,1,2]=4
        np.savez(p,reconstruction=x)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

def test_known_metric_case():
    x=np.zeros((1,2,3));x[0,1]=[1,2,.5];ref=x.copy();ref[0,1]=[.4,1.2,.25]
    raw={'edge_indices':np.array([[[0,1]]]),'relative_poses':np.zeros((1,1,3)),'information':np.diag([2,3,4])[None,None]}
    m=judge.score(x,ref,raw);assert m['position_max_error_m']==1.;assert m['heading_max_error_rad']==.25;assert m['weighted_observation_error']==15.

def test_asset_failure_truth_priority_and_invalid_boundary(tmp_path):
    a=tmp_path/'evaluation';a.mkdir();shutil.copytree(T/'data',tmp_path/'data');shutil.copy(T/'evaluation/metrics.json',a/'metrics.json')
    p=tmp_path/'pred.npz';np.savez(p,reconstruction=reference())
    assert judge.evaluate(p,a)[1]==2
    shutil.copy(T/'evaluation/baseline_reference.npz',a/'baseline_reference.npz');np.savez(a/'ground_truth.npz',reconstruction=np.zeros((1,1228,3)))
    assert judge.evaluate(p,a)[1]==1
    cfg=json.loads((a/'metrics.json').read_text());cfg['position_max_error_m_boundary']=float('nan');(a/'metrics.json').write_text(json.dumps(cfg));assert judge.evaluate(p,a)[1]==2

def test_cli_cwd_and_report_failure(tmp_path):
    p=tmp_path/'pred.npz';np.savez(p,reconstruction=reference());report=tmp_path/'deep/report.json'
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0;assert json.loads(report.read_text())['passed']
    blocker=tmp_path/'block';blocker.write_text('x');cmd[-1]=str(blocker/'report.json');assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
