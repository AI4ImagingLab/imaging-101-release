from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_flux_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_metric_known_case_and_zero_definition():
    r=np.array([1.,2.]);p=np.array([2.,1.])
    m=judge.measure(p,r)
    assert m['ncc_vs_ref']==pytest.approx(.8)
    assert m['nrmse_vs_ref']==pytest.approx(1.)
    assert judge.measure(np.zeros(2),r)['ncc_vs_ref']==0

@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','float16','nan','inf','broken'])
def test_invalid_submissions_have_empty_metrics(tmp_path,case):
    p=tmp_path/'prediction.npz'
    x=np.zeros(judge.SHAPE)
    if case=='missing':pass
    elif case=='broken':p.write_bytes(b'not an npz file')
    elif case=='key':np.savez(p,wrong=x)
    elif case=='shape':np.savez(p,reconstruction=x[...,0])
    elif case=='integer':np.savez(p,reconstruction=x.astype(np.int32))
    elif case=='complex':np.savez(p,reconstruction=x.astype(np.complex128))
    elif case=='float16':np.savez(p,reconstruction=x.astype(np.float16))
    else:
        x.flat[0]=np.nan if case=='nan' else np.inf
        np.savez(p,reconstruction=x)
    report,code=judge.evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={}
    assert not report['passed'] and report['errors']
    json.dumps(report,allow_nan=False)

def test_reference_pass_zero_fails_and_evaluator_assets_fail(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==0 and report['passed']
    p=tmp_path/'zero.npz';np.savez_compressed(p,reconstruction=np.zeros(judge.SHAPE))
    report,code=judge.evaluate(p)
    assert code==1 and not report['passed'] and report['status']=='fail'
    report,code=judge.evaluate(p,asset_dir=tmp_path/'missing_assets')
    assert code==2 and report['metrics']=={} and 'Evaluator assets' in report['errors'][0]

def test_cli_foreign_cwd_and_report_write_failure(tmp_path):
    script=ROOT/'evaluation/judge.py'
    report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(script),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)]
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(report.read_text())['passed']
    report.parent.mkdir(exist_ok=True)
    bad=tmp_path/'directory';bad.mkdir()
    cmd[-1]=str(bad)
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==2 and 'Could not write' in p.stderr

def test_constant_evaluator_reference_rejected(tmp_path):
    np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=np.ones(judge.SHAPE))
    (tmp_path/'metrics.json').write_text(json.dumps({'ncc_boundary':.9,'nrmse_boundary':.1}))
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',asset_dir=tmp_path)
    assert code==2 and report['metrics']=={}
