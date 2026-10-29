from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.physics_model import parametric_pdf,roughness,mass
from src.solvers import reconstruct
from src.visualization import comparison_metrics
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py'); judge=importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)
def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f: return {k:f[k].copy() for k in f.files}
def run(tmp_path,arrays):
    path=tmp_path/'p.npz'; np.savez(path,**arrays)
    report=tmp_path/'nested/report.json'
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True)
    return proc.returncode,json.loads(report.read_text())
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_good_dtypes_external_cwd(tmp_path,dtype):
    code,r=run(tmp_path,{k:v.astype(dtype) for k,v in valid().items()}); assert code==0 and r['passed']
@pytest.mark.parametrize('mode',['missing_key','shape','float16','integer','complex','nan','inf','huge','negative','zero_roughness'])
def test_invalid(tmp_path,mode):
    a=valid()
    if mode=='missing_key': del a['roughness']
    elif mode=='shape': a['reconstruction']=a['reconstruction'][0]
    elif mode in ['float16','integer','complex']: a['reconstruction']=a['reconstruction'].astype({'float16':np.float16,'integer':np.int64,'complex':np.complex128}[mode])
    elif mode=='zero_roughness': a['roughness']*=0
    else: a['reconstruction'][0,0]={'nan':np.nan,'inf':np.inf,'huge':1e308,'negative':-1}[mode]
    code,r=run(tmp_path,a); assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['flat','reversed','roughness','scaled'])
def test_poor(tmp_path,kind):
    a=valid()
    if kind=='flat': a['reconstruction'][:]=1
    elif kind=='reversed': a['reconstruction']=a['reconstruction'][:,::-1]
    elif kind=='roughness': a['roughness']*=2
    else: a['reconstruction']*=2
    a['reconstruction'][0,-1]=0
    code,r=run(tmp_path,a); assert code==1 and not r['passed']
def test_missing_file(tmp_path):
    out=tmp_path/'report.json'; p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none.npz'),'--report',str(out)],capture_output=True)
    assert p.returncode==2 and json.loads(out.read_text())['metrics']=={}
def test_missing_reference(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text()); monkeypatch.setattr(judge,'HERE',tmp_path)
    with pytest.raises(FileNotFoundError): judge.evaluate(ROOT/'output/reconstruction.npz')
def test_report_write_failure(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
def test_metric_known_case():
    a=np.array([1.,2.,3.]); m=comparison_metrics(a*2,a)
    assert m['ncc']==pytest.approx(1) and m['nrmse']==pytest.approx(np.sqrt(14/3)/2)
def test_analytic_distribution_mass():
    from scipy.integrate import quad
    for family in ['beckmann','ggx']:
        v=quad(lambda s:2*np.pi*s*parametric_pdf(s,.3,family),0,np.inf)[0]
        assert abs(v-1)<1e-10
def test_invalid_iterations():
    with pytest.raises(ValueError): reconstruct(None,0)
def test_nonzero_horizon(tmp_path):
    a=valid(); a['reconstruction'][0,-1]=1e-15
    code,r=run(tmp_path,a)
    assert code==2 and r['metrics']=={} and 'Horizon' in r['errors'][0]
