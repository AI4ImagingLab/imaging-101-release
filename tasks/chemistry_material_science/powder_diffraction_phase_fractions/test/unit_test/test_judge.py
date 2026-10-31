from pathlib import Path
import json,subprocess,sys,importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py'); judge=importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)
def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f: return {k:f[k].copy() for k in f.files}
def run(tmp_path,values):
    p=tmp_path/'prediction.npz'; np.savez(p,**values); report=tmp_path/'nested/report.json'
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    return proc.returncode,json.loads(report.read_text())
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_valid_external_cwd(tmp_path,dtype):
    code,r=run(tmp_path,{k:v.astype(dtype) for k,v in valid().items()}); assert code==0 and r['passed']
@pytest.mark.parametrize('kind',['missing','shape','float16','integer','complex','nan','inf','huge','negative','standard','shift'])
def test_invalid(tmp_path,kind):
    a=valid()
    if kind=='missing': del a['shifts']
    elif kind=='shape': a['fitted']=a['fitted'][0]
    elif kind in ['float16','integer','complex']: a['fitted']=a['fitted'].astype({'float16':np.float16,'integer':np.int64,'complex':np.complex128}[kind])
    elif kind=='standard': a['reconstruction'][0,0,0]=19
    elif kind=='shift': a['shifts'][0,0]=.4
    else: a['fitted'][0,0,0]={'nan':np.nan,'inf':np.inf,'huge':1e308,'negative':-1}[kind]
    code,r=run(tmp_path,a); assert code==2 and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['fractions','profile','shift'])
def test_each_required_output_scored(tmp_path,kind):
    a=valid()
    if kind=='fractions': a['reconstruction'][0,:,1:]=10
    elif kind=='profile': a['fitted']*=1.2
    else: a['shifts']-=.01
    code,r=run(tmp_path,a); assert code==1 and not r['passed']
def test_missing_submission(tmp_path):
    out=tmp_path/'report.json'; p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(out)],capture_output=True)
    assert p.returncode==2 and json.loads(out.read_text())['metrics']=={}
def test_missing_assets(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text()); monkeypatch.setattr(judge,'HERE',tmp_path)
    with pytest.raises(FileNotFoundError): judge.evaluate(ROOT/'output/reconstruction.npz')
def test_report_failure(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
