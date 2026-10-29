from pathlib import Path
import importlib.util,sys,subprocess
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics():
    a=np.array([0.,1.,2.]);assert judge.metrics(a,a)==dict(ncc=1.,nrmse=0.) or np.isclose(judge.metrics(a,a)['ncc'],1)
    assert np.isclose(judge.metrics(np.zeros(3),a)['nrmse'],np.sqrt(5/3)/2)

@pytest.mark.parametrize('kind',['missing','key','shape','complex','integer','float16','nan','inf'])
def test_invalid(tmp_path,kind):
    path=tmp_path/'bad.npz'
    if kind!='missing':
        a=np.zeros(judge.SHAPE)
        if kind=='shape':a=a[:,:,:1]
        if kind in ('complex','integer','float16'):a=a.astype({'complex':complex,'integer':int,'float16':np.float16}[kind])
        if kind in ('nan','inf'):a.flat[0]=float(kind)
        np.savez_compressed(path,**{('wrong' if kind=='key' else 'reconstruction'):a})
    r=judge.evaluate(path);assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_poor_and_reference(tmp_path):
    z=np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False)['reconstruction']
    good=tmp_path/'good.npz';bad=tmp_path/'bad.npz';np.savez_compressed(good,reconstruction=z);np.savez_compressed(bad,reconstruction=np.zeros_like(z))
    assert judge.evaluate(good)['passed'];assert judge.evaluate(bad)['status']=='fail'
    assert judge.evaluate(good,tmp_path)['status']=='error'

def test_cli_errors(tmp_path):
    report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction','absent.npz','--report',str(report)],cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and report.is_file()
    p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction','absent.npz','--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
