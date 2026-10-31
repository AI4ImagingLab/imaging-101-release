from pathlib import Path
import json,importlib.util,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import load_raw,fit_surface

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def evaluator(tmp_path):
    root=tmp_path/'evaluation';root.mkdir()
    target=np.linspace(-3.,7.,3166)[None]
    np.savez_compressed(root/'baseline_reference.npz',reconstruction=target)
    cfg={'ncc_boundary':.99,'nrmse_boundary':.01,'max_abs_boundary':.1}
    (root/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    return root,target,cfg

def test_independently_known_metric_formulas():
    a=np.array([1.,2.,3.]);b=np.array([2.,2.,4.])
    got=judge.field_metrics(a,b)
    assert got['ncc']==pytest.approx(18/np.sqrt(14*24))
    assert got['nrmse']==pytest.approx(np.sqrt(2/3)/2)
    assert got['max_abs_mpa']==1.
    assert judge.field_metrics(np.zeros(3),b)['ncc']==0
    with pytest.raises(ValueError):judge.field_metrics(a,np.ones(3))

@pytest.mark.parametrize('kind',['missing','wrong_key','extra_key','wrong_shape','float32','int','complex','object','nan','inf'])
def test_reject_malformed_prediction(evaluator,tmp_path,kind):
    root,target,_=evaluator;path=tmp_path/'prediction.npz';key='reconstruction';a=target.copy()
    if kind=='wrong_key':key='other'
    if kind=='wrong_shape':a=a.ravel()
    if kind=='float32':a=a.astype(np.float32)
    if kind=='int':a=a.astype(np.int64)
    if kind=='complex':a=a.astype(complex)
    if kind=='object':a=np.array([['bad']],dtype=object)
    if kind=='nan':a[0,0]=np.nan
    if kind=='inf':a[0,0]=np.inf
    if kind!='missing':np.savez_compressed(path,**{key:a},**({'extra':a} if kind=='extra_key' else {}))
    report=judge.evaluate(path,root)
    assert report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('kind',['zero','constant','sign','one_node'])
def test_reject_bad_valid_fields(evaluator,tmp_path,kind):
    root,target,_=evaluator;a=target.copy()
    if kind=='zero':a[:]=0
    if kind=='constant':a[:]=2
    if kind=='sign':a*=-1
    if kind=='one_node':a[0,10]+=1
    path=tmp_path/'bad.npz';np.savez_compressed(path,reconstruction=a)
    r=judge.evaluate(path,root);assert r['status']=='fail' and not r['passed'] and r['metrics']

@pytest.mark.parametrize('bad',[None,True,-1.,float('nan'),float('inf'),'0.1'])
def test_reject_invalid_configuration(evaluator,tmp_path,bad):
    root,target,cfg=evaluator;cfg['nrmse_boundary']=bad
    (root/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    p=tmp_path/'ok.npz';np.savez_compressed(p,reconstruction=target)
    r=judge.evaluate(p,root);assert r['status']=='error' and r['metrics']=={}

def test_truth_priority_and_no_invalid_truth_fallback(evaluator,tmp_path):
    root,target,_=evaluator;p=tmp_path/'truth.npz'
    np.savez_compressed(root/'ground_truth.npz',reconstruction=target*2)
    np.savez_compressed(p,reconstruction=target*2)
    assert judge.evaluate(p,root)['status']=='pass'
    np.savez_compressed(root/'ground_truth.npz',wrong=target)
    assert judge.evaluate(p,root)['status']=='error'

def test_missing_assets_and_finite_overflow(evaluator,tmp_path):
    root,target,_=evaluator;p=tmp_path/'huge.npz'
    np.savez_compressed(p,reconstruction=np.full_like(target,1e308))
    r=judge.evaluate(p,root);assert r['status']=='error' and r['metrics']=={}
    (root/'baseline_reference.npz').unlink()
    assert judge.evaluate(p,root)['status']=='error'

def test_standalone_cli_exit_codes_and_report_write_failure(tmp_path):
    for kind,code in [('missing',2),('poor',1),('good',0)]:
        p=tmp_path/(kind+'.npz');report=tmp_path/(kind+'.json')
        if kind=='poor':np.savez_compressed(p,reconstruction=np.zeros((1,3166)))
        if kind=='good':p.write_bytes((ROOT/'evaluation/baseline_reference.npz').read_bytes())
        r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
        assert r.returncode==code and report.is_file()
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'good.npz'),'--report',str(tmp_path)],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and b'Report-write failure' in r.stderr

def test_public_metadata_matches_runtime_and_no_hidden_loader():
    raw=load_raw(ROOT/'data/raw_data.npz');assert int(raw['fit_mask'].sum())==1917
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    assert '195600' in readme and '0.294' in readme and '3166' in readme
    for k in raw:assert '| '+k+' |' in readme
    assert 'github' not in readme.lower()
    with pytest.raises(ValueError):fit_surface(raw,(100,100))
