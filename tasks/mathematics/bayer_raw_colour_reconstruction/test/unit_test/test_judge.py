from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('standalone_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(s);s.loader.exec_module(judge)

def assets(tmp):
    e=tmp/'evaluation';e.mkdir();(tmp/'data').mkdir()
    a=np.arange(6*8*3,dtype=np.float64).reshape(1,6,8,3)/992
    raw=np.zeros((1,6,8),dtype=np.uint16)
    for r,c,ch in [(0,0,0),(0,1,1),(1,0,1),(1,1,2)]:raw[:,r::2,c::2]=np.rint(a[:,r::2,c::2,ch]*992+31).astype(np.uint16)
    np.savez(tmp/'data/raw_data.npz',sensor_mosaic=raw,black_levels=np.array([[31.]*4]))
    np.savez(e/'baseline_reference.npz',reconstruction=a)
    cfg={'outputs':{'reconstruction':{'shape':list(a.shape),'dtype':'float64'}},'ncc_boundary':.999,'nrmse_boundary':1e-8,'sample_consistency_max_abs_boundary':1e-12,'acquisition':{'pattern':'RGGB','white_level':1023}}
    (e/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    return e,a

def test_known_metric_and_undefined_reference():
    m=judge.image_metrics(np.array([1.,0]),np.array([0.,1]))
    assert m=={'ncc':0.,'nrmse':1.}
    with pytest.raises(ValueError):judge.image_metrics(np.ones(2),np.ones(2))

@pytest.mark.parametrize('kind',['good','poor','missing_key','wrong_shape','wrong_dtype','nan','inf','overflow','missing_file'])
def test_quality_and_invalid_contract(tmp_path,kind):
    e,a=assets(tmp_path);p=tmp_path/'prediction.npz'
    if kind=='poor':a=np.zeros_like(a)
    elif kind=='wrong_shape':a=a[:,1:]
    elif kind=='wrong_dtype':a=a.astype(np.float32)
    elif kind=='nan':a.flat[0]=np.nan
    elif kind=='inf':a.flat[0]=np.inf
    elif kind=='overflow':a.fill(1e300)
    if kind!='missing_file':np.savez(p,**({'other':a} if kind=='missing_key' else {'reconstruction':a}))
    report,code=judge.evaluate(p,e)
    assert code==(0 if kind=='good' else 1 if kind=='poor' else 2)
    assert report['passed']==(kind=='good')
    if code==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

def test_truth_priority_and_evaluator_missing_asset(tmp_path):
    e,a=assets(tmp_path);p=tmp_path/'p.npz';np.savez(p,reconstruction=a)
    np.savez(e/'baseline_reference.npz',reconstruction=np.zeros_like(a))
    np.savez(e/'ground_truth.npz',reconstruction=a)
    assert judge.evaluate(p,e)[1]==0
    (e/'ground_truth.npz').unlink();(e/'baseline_reference.npz').unlink()
    assert judge.evaluate(p,e)[1]==2

def test_cli_foreign_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(tmp_path/'absent.npz'),'--report',str(tmp_path/'nested/report.json')]
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==2 and json.loads((tmp_path/'nested/report.json').read_text())['metrics']=={}
    p=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==2 and 'Cannot write report' in p.stderr
