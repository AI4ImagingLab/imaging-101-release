from pathlib import Path
import importlib.util
import subprocess
import sys
import json
import numpy as np
import pytest
from src.visualization import similarity

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
SHAPE=(1,25,64,48)

def test_known_numerical_metrics():
    a=np.array([1.,2.,3.]);b=np.array([2.,2.,4.]);expected=(18/np.sqrt(14*24),np.sqrt(2/3)/2)
    np.testing.assert_allclose(judge.similarity(a,b),expected,rtol=1e-15)
    np.testing.assert_allclose(similarity(a,b),expected,rtol=1e-15)

@pytest.mark.parametrize('kind',['missing_key','shape','float32','complex','nan','inf','overflow'])
def test_invalid_predictions(tmp_path,kind):
    a=np.ones(SHAPE);key='reconstruction'
    if kind=='missing_key':key='other'
    if kind=='shape':a=a[0]
    if kind=='float32':a=a.astype('float32')
    if kind=='complex':a=a.astype(complex)
    if kind=='nan':a.flat[0]=np.nan
    if kind=='inf':a.flat[0]=np.inf
    if kind=='overflow':a[:]=1e308
    path=tmp_path/'bad.npz';np.savez(path,**{key:a})
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={} and report['status']=='error' and report['errors']

def test_missing_file_bad_archive_and_quality(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')[1]==2
    p=tmp_path/'invalid.npz';p.write_bytes(b'invalid archive');assert judge.evaluate(p)[1]==2
    np.savez(p,reconstruction=np.zeros(SHAPE));report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail' and np.isfinite(list(report['metrics'].values())).all()

def test_evaluator_assets_and_invalid_thresholds(tmp_path):
    assets=tmp_path/'evaluation';assets.mkdir()
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['ncc_boundary']='invalid'
    (assets/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2
    config['ncc_boundary']=.9;(assets/'metrics.json').write_text(json.dumps(config))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2
    np.savez(assets/'baseline_reference.npz',reconstruction=np.zeros(SHAPE))
    assert judge.evaluate(ROOT/'output/reconstruction.npz',assets)[1]==2

def test_cli_foreign_cwd_and_report_parent(tmp_path):
    p=tmp_path/'prediction.npz';r=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,'-m','src.main','--output',str(p)],cwd=ROOT,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert json.loads(r.read_text())['passed'] is True

def test_report_write_failure_and_backend(tmp_path):
    directory=tmp_path/'directory';directory.mkdir()
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(directory)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Unable to write report' in result.stderr
    import matplotlib
    backend=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==backend
