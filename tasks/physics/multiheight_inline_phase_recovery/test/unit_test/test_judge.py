import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
def run(pred,report,cwd):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=cwd,capture_output=True,text=True)
    return p,json.loads(report.read_text())
@pytest.mark.parametrize('case',['missing_file','missing_key','shape','dtype','nan','inf'])
def test_malformed(tmp_path,case):
    a=np.zeros((1,512,512),complex);p=tmp_path/'prediction.npz'
    if case=='missing_key':np.savez(p,wrong=a)
    elif case!='missing_file':
        if case=='shape':a=a[:,:,:1]
        if case=='dtype':a=a.real
        if case=='nan':a[0,0,0]=complex(float('nan'),0)
        if case=='inf':a[0,0,0]=complex(float('inf'),0)
        np.savez(p,reconstruction=a)
    proc,r=run(p,tmp_path/'report.json',tmp_path);assert proc.returncode==2;assert r['status']=='error';assert r['metrics']=={}
@pytest.mark.parametrize('extreme',[False,True])
def test_quality_failure_including_overflow(tmp_path,extreme):
    a=np.full((1,512,512),1e308+1e308j if extreme else 0j,dtype='complex128');p=tmp_path/'prediction.npz';np.savez(p,reconstruction=a)
    proc,r=run(p,tmp_path/'report.json',tmp_path);assert proc.returncode==1;assert r['status']=='fail';assert all(np.isfinite(v) for v in r['metrics'].values())
def test_reference_passes_from_foreign_cwd(tmp_path):
    p=ROOT/'evaluation/baseline_reference.npz';proc,r=run(p,tmp_path/'nested/report.json',tmp_path);assert proc.returncode==0,r;assert r['passed']
def test_evaluator_asset_failure(tmp_path):
    isolated=tmp_path/'evaluation';isolated.mkdir();(isolated/'judge.py').write_bytes((ROOT/'evaluation/judge.py').read_bytes())
    proc=subprocess.run([sys.executable,str(isolated/'judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'report.json')],capture_output=True)
    r=json.loads((tmp_path/'report.json').read_text());assert proc.returncode==2;assert r['metrics']=={}
def test_report_failure(tmp_path):
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path)],capture_output=True)
    assert proc.returncode==2;assert b'report write failed' in proc.stderr
