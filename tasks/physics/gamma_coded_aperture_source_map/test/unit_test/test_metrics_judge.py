from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.visualization import metrics,contrast_to_noise
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_known_metric_case():
    x=np.array([1.,0.]);r=np.array([1.,2.]);expected={'ncc':1/np.sqrt(5),'nrmse':np.sqrt(2.)}
    for k,v in expected.items():assert np.isclose(metrics(x,r)[k],v);assert np.isclose(judge.score(x,r)[k],v)
    assert judge.score(np.zeros(2),r)['ncc']==0
def test_diagnostic_contrast_is_explicit_roi():
    a=np.indices((15,15)).sum(axis=0).astype(float)%2;a[6:9,6:9]=4;a[7,7]=5
    rr,cc=np.indices(a.shape);bg=(abs(rr-7)>5)|(abs(cc-7)>5)
    expected=((4*8+5)/9-a[bg].mean())/a[bg].std()
    assert np.isclose(contrast_to_noise(a),expected)
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','overflow'])
def test_bad_prediction(kind,tmp_path):
    p=tmp_path/'x.npz';a=np.ones((1,126,126),np.float32)
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=a)
    else:
        if kind=='shape':a=a[:,1:]
        if kind=='dtype':a=a.astype(np.float64)
        if kind=='complex':a=a.astype(np.complex64)
        if kind in ['nan','inf']:a.flat[0]=float(kind)
        if kind=='overflow':a.fill(np.finfo(np.float32).max)
        np.savez(p,reconstruction=a)
    report,code=judge.evaluate(p)
    if kind=='overflow':assert code==1
    else:assert code==2 and report['metrics']=={} and report['errors']
def test_reference_pass_and_zero_fail(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz');assert code==0 and report['passed']
    p=tmp_path/'zero.npz';np.savez(p,reconstruction=np.zeros((1,126,126),np.float32));report,code=judge.evaluate(p);assert code==1 and not report['passed']
def test_hidden_assets_and_configuration_errors(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'ASSETS',tmp_path);report,code=judge.evaluate('anything');assert code==2 and report['metrics']=={}
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":0,"nrmse_boundary":0}');assert judge.evaluate('anything')[1]==2
def test_cli_report_write_error(tmp_path):
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path)],capture_output=True,text=True);assert r.returncode==2 and 'Cannot write report' in r.stderr
