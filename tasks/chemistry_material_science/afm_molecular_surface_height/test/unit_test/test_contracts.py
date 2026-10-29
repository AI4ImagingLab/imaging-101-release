import importlib.util,json,os,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pytest
from src.visualization import metrics
from src.preprocessing import load_observations,PIXEL_PITCH,NOISE_SIGMA
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def run_judge(pred,dest,script=None):
    p=subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(dest)],capture_output=True,text=True)
    return p

def test_metrics_independently_known_case():
    r=np.array([0.,1.,2.]);x=np.array([0.,2.,4.]);m=metrics(x,r);j=judge.score(x,r)
    assert abs(m['ncc']-1)<1e-14 and abs(m['nrmse']-np.sqrt(5/3)/2)<1e-14
    assert m==j
    assert metrics(np.zeros(3),r)['ncc']==0

def test_constants_and_public_contract():
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    assert PIXEL_PITCH==15 and NOISE_SIGMA==3
    for s in ['15','3','(1, 30, 30, 30)','(1, 14, 14)','float64','output/reconstruction.npz']:assert s in readme
    forbidden=['github.com','githubusercontent.com','github.io','git.io','codeload.github.com']
    assert not any(s in readme.lower() for s in forbidden)
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:assert set(z.files)=={'measurements','probe'}

def test_standard_cli_and_judge_from_other_cwd(tmp_path):
    (tmp_path/'data').mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)
    a=subprocess.run([sys.executable,'-m','src.main'],cwd=tmp_path,env=env,capture_output=True,text=True);assert a.returncode==0,a.stderr
    pred=tmp_path/'output/reconstruction.npz';report=tmp_path/'reports/new.json';r=run_judge(pred,report);assert r.returncode==0,r.stderr
    j=json.loads(report.read_text());assert j['status']=='pass' and j['passed'] and not j['errors']

@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','nan','inf','object'])
def test_malformed_prediction_is_error(tmp_path,kind):
    path=tmp_path/'bad.npz';x=np.zeros((1,30,30,30),dtype=np.float64)
    if kind=='missing':pass
    elif kind=='key':np.savez(path,wrong=x)
    else:
        if kind=='shape':x=x[...,:-1]
        if kind=='float32':x=x.astype(np.float32)
        if kind=='complex':x=x.astype(np.complex128)
        if kind=='object':x=x.astype(object)
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        np.savez(path,reconstruction=x)
    report=tmp_path/'report.json';r=run_judge(path,report);assert r.returncode==2
    j=json.loads(report.read_text());assert j['status']=='error' and j['metrics']=={} and j['errors'] and not j['passed']

@pytest.mark.parametrize('scale',[0.,2.])
def test_poor_predictions_fail(tmp_path,scale):
    y,p=load_observations(ROOT/'data/raw_data.npz');x=reconstruct(y,p)*scale
    pred=tmp_path/'poor.npz';np.savez(pred,reconstruction=x[None]);report=tmp_path/'r.json';r=run_judge(pred,report)
    assert r.returncode==1
    j=json.loads(report.read_text());assert j['status']=='fail' and not j['passed'] and all(np.isfinite(list(j['metrics'].values())))

def test_missing_evaluator_asset_is_error(tmp_path):
    ev=tmp_path/'evaluation';ev.mkdir();shutil.copyfile(ROOT/'evaluation/judge.py',ev/'judge.py');shutil.copyfile(ROOT/'evaluation/metrics.json',ev/'metrics.json')
    p=tmp_path/'pred.npz';np.savez(p,reconstruction=np.zeros((1,30,30,30)));r=run_judge(p,tmp_path/'r.json',ev/'judge.py');assert r.returncode==2
    assert json.loads((tmp_path/'r.json').read_text())['metrics']=={}

def test_report_write_failure(tmp_path):
    pred=tmp_path/'pred.npz';np.savez(pred,reconstruction=np.zeros((1,30,30,30)))
    dest=tmp_path/'directory';dest.mkdir();r=run_judge(pred,dest);assert r.returncode==2 and 'Cannot write report' in r.stderr

def test_import_main_does_not_change_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before

@pytest.mark.parametrize('key,value',[
    ('ncc_boundary',-1.01),('ncc_boundary',1.01),('ncc_boundary',True),
    ('ncc_boundary',float('nan')),('ncc_boundary',float('inf')),
    ('nrmse_boundary',-0.01),('nrmse_boundary',False),
    ('observation_rmse_boundary',-1.),('observation_rmse_boundary','3')])
def test_invalid_hidden_threshold_is_error(tmp_path,monkeypatch,key,value):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    config[key]=value
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    monkeypatch.setattr(judge,'ASSETS',tmp_path)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz')
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_finite_extreme_prediction_produces_serializable_error(tmp_path):
    pred=tmp_path/'extreme.npz'
    np.savez(pred,reconstruction=np.full((1,30,30,30),1e308,dtype=np.float64))
    dest=tmp_path/'report.json';result=run_judge(pred,dest)
    assert result.returncode==2
    report=json.loads(dest.read_text())
    assert report['status']=='error' and report['metrics']=={} and report['errors']
