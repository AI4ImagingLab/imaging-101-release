import importlib.util,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
J=Path('evaluation/judge.py').resolve()
spec=importlib.util.spec_from_file_location('standalone_judge',J);judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_metrics_independent_known_cases():
    assert judge.observation_vr(np.zeros((3,3)),10.)==0
    ref=np.load('evaluation/baseline_reference.npz')['reconstruction'][0]
    assert abs(judge.observation_vr(ref,10.)-74.46893184453873)<1e-10

@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','asymmetric','depth','poor'])
def test_prediction_rejections(tmp_path,case):
    p=tmp_path/'prediction.npz';out=tmp_path/'report/report.json'
    with np.load('evaluation/baseline_reference.npz') as z:arrays={k:z[k].copy() for k in z.files}
    if case=='key':arrays.pop('reconstruction')
    if case=='shape':arrays['reconstruction']=np.ones((3,3),dtype=np.float64)
    if case=='dtype':arrays['reconstruction']=arrays['reconstruction'].astype(np.float32)
    if case=='complex':arrays['reconstruction']=arrays['reconstruction'].astype(np.complex128)
    if case in ('nan','inf'):arrays['reconstruction'][0,0,0]=float(case)
    if case=='asymmetric':arrays['reconstruction'][0,0,1]+=1e13
    if case=='depth':arrays['source_depth'][0,0]=11.
    if case=='poor':arrays['reconstruction'][:]=0
    if case!='missing':np.savez(p,**arrays)
    proc=subprocess.run([sys.executable,str(J),'--prediction',str(p),'--report',str(out)],capture_output=True,text=True,cwd=tmp_path)
    assert proc.returncode==(1 if case=='poor' else 2)
    report=json.loads(out.read_text());assert not report['passed']
    if case!='poor':assert report['metrics']=={} and report['errors']

def test_evaluator_and_report_write_failure(tmp_path):
    other=tmp_path/'judge.py';shutil.copy2(J,other)
    out=tmp_path/'error.json'
    proc=subprocess.run([sys.executable,str(other),'--prediction',str(Path('output/reconstruction.npz').resolve()),'--report',str(out)],capture_output=True,text=True)
    assert proc.returncode==2 and json.loads(out.read_text())['metrics']=={}
    blocked=tmp_path/'file';blocked.write_text('parent is file')
    proc=subprocess.run([sys.executable,str(J),'--prediction','output/reconstruction.npz','--report',str(blocked/'report.json')],capture_output=True,text=True)
    assert proc.returncode==2 and 'Cannot write report' in proc.stderr
