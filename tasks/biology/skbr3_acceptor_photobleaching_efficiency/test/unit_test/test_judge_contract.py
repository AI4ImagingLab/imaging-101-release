from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_fret_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def assets(tmp_path,monkeypatch):
    e=tmp_path/'evaluation';e.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'evaluation/metrics.json',e/'metrics.json')
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',e/'baseline_reference.npz')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as a:mask=a['observation_mask']
    np.savez(tmp_path/'data/raw_data.npz',observation_mask=mask)
    monkeypatch.setattr(judge,'HERE',e)
    with np.load(e/'baseline_reference.npz',allow_pickle=False) as a:ref={k:a[k].copy() for k in a.files}
    return tmp_path,e,ref

def save_and_score(assets,values):
    root,_,_=assets;path=root/'prediction.npz';np.savez(path,**values);return judge.evaluate(path)

def test_valid_reference_pass(assets):assert save_and_score(assets,assets[2])['status']=='pass'

@pytest.mark.parametrize('kind',['missing_key','extra_key','wrong_shape','integer','complex','nan','inf','outside'])
def test_prediction_contract_errors(assets,kind):
    values=assets[2]
    if kind=='missing_key':values.pop('correction_factors')
    elif kind=='extra_key':values['unexpected']=np.ones(1)
    elif kind=='wrong_shape':values['reconstruction']=values['reconstruction'][0]
    elif kind=='integer':values['reconstruction']=values['reconstruction'].astype(np.int16)
    elif kind=='complex':values['reconstruction']=values['reconstruction'].astype(complex)
    elif kind in ['nan','inf']:values['correction_factors'][0,0]=np.nan if kind=='nan' else np.inf
    else:
        with np.load(assets[0]/'data/raw_data.npz',allow_pickle=False) as a:outside=tuple(np.argwhere(~a['observation_mask'])[0])
        values['reconstruction'][outside]=1
    r=save_and_score(assets,values);assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_zero_valid_field_fails_without_nan(assets):
    values=assets[2];values['reconstruction']*=0
    r=save_and_score(assets,values)
    assert r['status']=='fail' and not r['passed'] and r['metrics']['efficiency_ncc']==0
    json.dumps(r,allow_nan=False)

def test_every_factor_is_scored(assets):
    values=assets[2];values['correction_factors'][0,3]+=.01
    assert save_and_score(assets,values)['status']=='fail'

def test_actual_truth_priority_and_invalid_truth_no_fallback(assets):
    _,e,values=assets
    truth={k:v.copy() for k,v in values.items()};truth['reconstruction']*=2
    np.savez(e/'ground_truth.npz',**truth)
    assert save_and_score(assets,values)['status']=='fail'
    assert save_and_score(assets,truth)['status']=='pass'
    np.savez(e/'ground_truth.npz',reconstruction=np.ones(1))
    r=save_and_score(assets,values);assert r['status']=='error' and r['metrics']=={}

def test_missing_prediction_and_reference_errors(assets):
    root,e,_=assets
    assert judge.evaluate(root/'absent.npz')['status']=='error'
    (e/'baseline_reference.npz').unlink()
    assert judge.evaluate(root/'absent.npz')['errors'][0].startswith('Evaluator error:')

@pytest.mark.parametrize('bad',[float('nan'),True,-1])
def test_invalid_metric_assets_rejected(assets,bad):
    _,e,values=assets;cfg=json.loads((e/'metrics.json').read_text());cfg['nrmse_boundary']=bad
    (e/'metrics.json').write_text(json.dumps(cfg))
    assert save_and_score(assets,values)['status']=='error'

def test_foreign_cwd_cli_and_report_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report','nested/report.json']
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed']
    (tmp_path/'blocked').write_text('file')
    p=subprocess.run(command[:-1]+['blocked/report.json'],cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
