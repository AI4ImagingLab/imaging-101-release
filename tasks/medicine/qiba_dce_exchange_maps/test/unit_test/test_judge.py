from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_qiba_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def analytic_fields():
    x=np.linspace(.01,.35,3000).reshape(1,60,50)
    return {'ktrans':x,'extracellular_fraction':x+.1}

@pytest.fixture
def assets(tmp_path):
    folder=tmp_path/'evaluator';folder.mkdir()
    np.savez(folder/'ground_truth.npz',**analytic_fields())
    config={'quantities':{k:{'ncc_boundary':.999,'nrmse_boundary':.01} for k in judge.KEYS}}
    (folder/'metrics.json').write_text(json.dumps(config),encoding='utf8')
    return folder

def test_analytic_metrics():
    ncc,nrmse=judge.compare(np.array([0.,2.]),np.array([1.,2.]))
    assert ncc==pytest.approx(2/np.sqrt(5))
    assert nrmse==pytest.approx(1/np.sqrt(2))
    assert judge.compare(np.zeros(2),np.array([1.,2.]))[0]==0

def test_truth_priority_and_broken_truth_no_fallback(assets,tmp_path):
    truth=analytic_fields();wrong={k:np.flip(v,axis=2).copy() for k,v in truth.items()}
    np.savez(assets/'baseline_reference.npz',**wrong)
    pred=tmp_path/'prediction.npz';np.savez(pred,**truth)
    report,code=judge.evaluate(pred,assets)
    assert code==0 and report['passed']
    assert all(v==pytest.approx(0 if k.endswith('nrmse') else 1) for k,v in report['metrics'].items())
    (assets/'ground_truth.npz').write_bytes(b'corrupt actual truth')
    report,code=judge.evaluate(pred,assets)
    assert code==2 and not report['metrics'] and 'Evaluator assets' in report['errors'][0]

def test_method_reference_only_when_truth_absent(assets,tmp_path):
    (assets/'ground_truth.npz').rename(assets/'baseline_reference.npz')
    pred=tmp_path/'prediction.npz';np.savez(pred,**analytic_fields())
    assert judge.evaluate(pred,assets)[1]==0

@pytest.mark.parametrize('variant',['missing','wrong_key','extra_key','shape','integer','float16','complex','nan','inf','object','negative','fraction_cap','ktrans_cap','invalid_zip'])
def test_malformed_predictions(variant,assets,tmp_path):
    pred=tmp_path/'prediction.npz';fields=analytic_fields()
    if variant=='missing':pass
    elif variant=='invalid_zip':pred.write_bytes(b'not an NPZ')
    else:
        if variant=='wrong_key':fields.pop('ktrans')
        elif variant=='extra_key':fields['extra']=np.ones(1)
        elif variant=='shape':fields['ktrans']=np.ones((60,50))
        elif variant in ['integer','float16','complex','object']:
            dtype={'integer':np.int32,'float16':np.float16,'complex':np.complex128,'object':object}[variant]
            fields['ktrans']=fields['ktrans'].astype(dtype)
        else:
            key='extracellular_fraction' if variant=='fraction_cap' else 'ktrans'
            value={'nan':np.nan,'inf':np.inf,'negative':-.1,'fraction_cap':1.1,'ktrans_cap':5.1}[variant]
            fields[key].flat[0]=value
        np.savez(pred,**fields)
    report,code=judge.evaluate(pred,assets)
    assert code==2 and report['status']=='error' and not report['metrics'] and report['errors']
    json.dumps(report,allow_nan=False)

def test_valid_poor_fields_fail_every_quantity(assets,tmp_path):
    pred=tmp_path/'prediction.npz';np.savez(pred,**{k:np.zeros(judge.SHAPE) for k in judge.KEYS})
    report,code=judge.evaluate(pred,assets)
    assert code==1 and report['status']=='fail' and len(report['metrics'])==4

@pytest.mark.parametrize('variant',['missing_reference','constant_reference','nan_reference','invalid_threshold','missing_quantity'])
def test_invalid_evaluator_assets(variant,assets,tmp_path):
    pred=tmp_path/'prediction.npz';np.savez(pred,**analytic_fields())
    if variant=='missing_reference':(assets/'ground_truth.npz').unlink()
    elif variant in ['constant_reference','nan_reference']:
        fields=analytic_fields();fields['ktrans'][:]=.2 if variant=='constant_reference' else np.nan
        np.savez(assets/'ground_truth.npz',**fields)
    else:
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        if variant=='invalid_threshold':cfg['quantities']['ktrans']['ncc_boundary']=float('nan')
        else:cfg['quantities'].pop('ktrans')
        (assets/'metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    report,code=judge.evaluate(pred,assets)
    assert code==2 and not report['metrics']

def test_foreign_cwd_cli_and_report_write_failure(tmp_path):
    prediction=tmp_path/'perfect.npz'
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as f:
        np.savez(prediction,**{k:f[k] for k in f.files})
    report=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,shell=False)
    assert result.returncode==0 and json.loads(report.read_text(encoding='utf8'))['passed']
    bad_parent=tmp_path/'ordinary_file';bad_parent.write_text('x',encoding='utf8')
    result=subprocess.run(command[:-1]+[str(bad_parent/'report.json')],cwd=tmp_path,capture_output=True,shell=False)
    assert result.returncode==2 and b'Cannot write' in result.stderr
