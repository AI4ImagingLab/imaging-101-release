import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge_contract',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def arrays():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


@pytest.mark.parametrize('case',['missing','key','shape','float32','complex','nan','inf','object','negative_se'])
def test_invalid_predictions(case,tmp_path):
    data=arrays();p=tmp_path/'bad.npz'
    if case=='missing':pass
    else:
        if case=='key':data.pop('reconstruction')
        elif case=='shape':data['reconstruction']=data['reconstruction'][:,1:]
        elif case=='float32':data['reconstruction']=data['reconstruction'].astype('float32')
        elif case=='complex':data['reconstruction']=data['reconstruction'].astype('complex128')
        elif case in ['nan','inf']:
            data['reconstruction']=data['reconstruction'].copy();data['reconstruction'][0,0]=float(case)
        elif case=='object':data['reconstruction']=data['reconstruction'].astype(object)
        elif case=='negative_se':
            data['standard_error']=data['standard_error'].copy();data['standard_error'][0,0]=-1
        np.savez(p,**data)
    r,code=judge.evaluate(p)
    assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']


def test_quality_failure_and_exact_reference_pass(tmp_path):
    p=tmp_path/'p.npz';data=arrays();np.savez(p,**data)
    r,code=judge.evaluate(p);assert code==0 and r['passed']
    np.savez(p,**{k:np.zeros_like(v) for k,v in data.items()})
    r,code=judge.evaluate(p);assert code==1 and r['status']=='fail' and r['metrics']
    for changed in [{k:2*v for k,v in data.items()},
                    {k:v[:,::-1] for k,v in data.items()}]:
        np.savez(p,**changed)
        r,code=judge.evaluate(p);assert code==1 and not r['passed']


def test_missing_and_degenerate_evaluator_assets(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**arrays())
    shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    r,code=judge.evaluate(p,tmp_path);assert code==2 and r['metrics']=={}
    np.savez(tmp_path/'baseline_reference.npz',**{k:np.ones_like(v) for k,v in arrays().items()})
    r,code=judge.evaluate(p,tmp_path);assert code==2 and 'Degenerate' in r['errors'][0]
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    config=json.loads((tmp_path/'metrics.json').read_text());config['reconstruction_nrmse_boundary']=0
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    r,code=judge.evaluate(p,tmp_path);assert code==2


def test_cli_caller_paths_nested_report_and_write_failure(tmp_path):
    np.savez(tmp_path/'submission.npz',**arrays())
    base=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction','submission.npz','--report']
    result=subprocess.run(base+['nested/report.json'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'nested/report.json').read_text())
    assert set(report)=={'status','passed','metrics','thresholds','errors'}
    assert not report['errors']
    result=subprocess.run(base+['.'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Cannot write' in result.stderr
