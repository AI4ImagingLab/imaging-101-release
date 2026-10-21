import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture(scope='module')
def valid():
    return judge.load_validated(ROOT/'evaluation/baseline_reference.npz')

def test_known_metrics():
    ncc,error=judge.pair_metrics(np.array([1.,2.,3.]),np.array([1.,2.,4.]))
    assert ncc==pytest.approx(17/np.sqrt(14*21))
    assert error==pytest.approx(np.sqrt(1/3)/3)
    from src.visualization import metrics
    plotted=metrics(np.array([1.,2.,3.]),np.array([1.,2.,4.]))
    assert plotted['ncc']==pytest.approx(ncc) and plotted['nrmse']==pytest.approx(error)

def test_truth_reference_is_not_invented():
    assert not (ROOT/'evaluation/ground_truth.npz').exists()

@pytest.mark.parametrize('case',['missing_key','extra_key','shape','dtype','complex','nan','inf','bad_count','bad_gauge','bad_innovations'])
def test_reject_malformed(tmp_path,valid,case):
    data={k:v.copy() for k,v in valid.items()}
    if case=='missing_key':data.pop('dff')
    elif case=='extra_key':data['extra']=np.zeros(1)
    elif case=='shape':data['fluorescence']=data['fluorescence'][...,:-1]
    elif case=='dtype':data['dff']=data['dff'].astype(np.int32)
    elif case=='complex':data['dff']=data['dff'].astype(np.complex128)
    elif case in ['nan','inf']:data['dff'].flat[0]=np.nan if case=='nan' else np.inf
    elif case=='bad_count':data['reconstruction']=data['reconstruction'][:,:0]
    elif case=='bad_gauge':data['reconstruction']*=2
    elif case=='bad_innovations':data['innovations'].flat[0]=-1
    path=tmp_path/'invalid.npz';np.savez_compressed(path,**data)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_missing_prediction_and_hidden_assets(tmp_path,valid):
    report,code=judge.evaluate(tmp_path/'missing.npz');assert code==2
    path=tmp_path/'valid.npz';np.savez_compressed(path,**valid)
    report,code=judge.evaluate(path,assets=tmp_path);assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('boundary,value',[('spatial_ncc_boundary',float('nan')),('background_ncc_boundary',float('inf')),('detection_f1_boundary',-0.1),('fluorescence_ncc_boundary',1.1),('spatial_nrmse_boundary',-1),('motion_max_error_boundary',-1),('dff_nrmse_boundary',True)])
def test_invalid_metric_configuration_is_evaluator_error(tmp_path,boundary,value):
    settings=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'));settings[boundary]=value
    (tmp_path/'metrics.json').write_text(json.dumps(settings),encoding='utf-8')
    report,code=judge.evaluate(tmp_path/'absent_prediction.npz',assets=tmp_path)
    assert code==2 and report['metrics']=={} and report['status']=='error'
    assert boundary in report['errors'][0]
    json.dumps(report,allow_nan=False)

def test_valid_poor_prediction_fails(tmp_path,valid):
    data={k:v.copy() for k,v in valid.items()};data['fluorescence']=np.ones_like(data['fluorescence'])
    path=tmp_path/'poor.npz';np.savez_compressed(path,**data)
    report,code=judge.evaluate(path);assert code==1 and not report['passed'] and report['metrics']

def test_zero_temporal_is_valid_but_fails(tmp_path,valid):
    data={k:v.copy() for k,v in valid.items()};data['fluorescence']=np.zeros_like(data['fluorescence'])
    path=tmp_path/'zero_temporal.npz';np.savez_compressed(path,**data)
    report,code=judge.evaluate(path);assert code==1 and report['metrics']['fluorescence_ncc']==0

def test_permutation_and_background_basis(valid):
    data={k:v.copy() for k,v in valid.items()}
    for key in ['reconstruction','fluorescence','innovations','dff']:data[key]=data[key][:,::-1]
    data['background_spatial']=data['background_spatial'][:,::-1];data['background_temporal']=data['background_temporal'][:,::-1]
    values=judge.score(data,valid)
    assert values['detection_f1']==1
    for key,v in values.items():assert v==pytest.approx(0 if key.endswith('nrmse') or key=='motion_max_error' else 1,abs=1e-12)

def test_report_write_error(tmp_path,capsys):
    directory=tmp_path/'directory';directory.mkdir()
    code=judge.main(['--prediction',str(tmp_path/'missing.npz'),'--report',str(directory)])
    assert code==2 and 'Cannot write report' in capsys.readouterr().err
