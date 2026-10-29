import importlib.util,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data,FS,KT
from src.visualization import relative_error,residual_rms
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def good():return np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
def test_constants_and_metrics():
    text=(ROOT/'README.md').read_text(encoding='utf8');assert '400' in text and '4.1' in text
    assert FS==400 and KT==4.1 and load_data(ROOT/'data/raw_data.npz').shape==(13631,)
    assert relative_error(np.array([2.,4.]),np.array([1.,2.]))==1
    assert residual_rms([3,4])==np.sqrt(12.5)
@pytest.mark.parametrize('case',['missing','key','shape','float16','integer','complex','nan','inf'])
def test_malformed(tmp_path,case):
    p=tmp_path/'x.npz';x=good()
    if case=='key':np.savez(p,wrong=x)
    elif case!='missing':
        if case=='shape':x=x[0]
        if case=='float16':x=x.astype(np.float16)
        if case=='integer':x=x.astype(int)
        if case=='complex':x=x.astype(complex)
        if case=='nan':x[0,0]=np.nan
        if case=='inf':x[0,0]=np.inf
        np.savez(p,reconstruction=x)
    r=judge.evaluate(p);assert r['status']=='error' and r['metrics']=={} and r['errors']
def test_poor_prediction(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,reconstruction=good()*2);assert judge.evaluate(p)['status']=='fail'
def test_actual_truth_priority_and_asset_errors(tmp_path):
    assets=tmp_path/'assets';shutil.copytree(ROOT/'evaluation',assets)
    p=tmp_path/'x.npz';np.savez(p,reconstruction=good())
    np.savez(assets/'ground_truth.npz',reconstruction=good()*2)
    assert judge.evaluate(p,assets)['metrics']['parameter_relative_error']==.5
    (assets/'ground_truth.npz').unlink();(assets/'baseline_reference.npz').unlink()
    assert judge.evaluate(p,assets)['status']=='error'
def test_pipeline_and_cli(tmp_path):
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True,capture_output=True)
    report=tmp_path/'nested/report.json'
    cp=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert cp.returncode==0 and json.loads(report.read_text())['passed']
    cp=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert cp.returncode==2 and b'Report write failed' in cp.stderr
