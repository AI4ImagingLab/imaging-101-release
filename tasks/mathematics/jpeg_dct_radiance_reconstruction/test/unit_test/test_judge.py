from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.mark.parametrize('case',['missing_key','shape','integer','complex','nan','inf','missing','not_npz'])
def test_invalid_inputs(tmp_path,case):
    p=tmp_path/'prediction.npz';u=np.zeros((1,3,432,640),dtype=np.float32)
    if case=='missing':pass
    elif case=='not_npz':p.write_bytes(b'invalid')
    elif case=='missing_key':np.savez(p,wrong=u)
    else:
        if case=='shape':u=u[...,:1]
        elif case=='integer':u=u.astype(np.int16)
        elif case=='complex':u=u.astype(np.complex64)
        elif case=='nan':u.flat[0]=np.nan
        elif case=='inf':u.flat[0]=np.inf
        np.savez(p,reconstruction=u)
    report,code=judge.evaluate(p)
    assert code==2 and report['status']=='error' and not report['passed']
    assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)
def test_valid_bad_outputs_fail(tmp_path):
    for label,u in [('zero',np.zeros((1,3,432,640),dtype=np.float32)),('midpoint',np.load(ROOT/'test/fixtures/upstream_states.npz')['output_decode'][None])]:
        p=tmp_path/(label+'.npz');np.savez(p,reconstruction=u)
        report,code=judge.evaluate(p)
        assert code==1 and report['status']=='fail' and report['metrics']
def test_known_constant_score():
    u=np.ones((3,432,640));q=np.ones((3,8,8),dtype=np.uint16)
    a=np.zeros((3,54,80,8,8),dtype=np.int16);a[...,0,0]=8
    values=judge.score(u,a,q)
    assert values['objective']<1e-20 and values['coefficient_violation']<1e-12
def test_cli_unrelated_cwd_and_report_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    process=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert process.returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    command[-1]=str(tmp_path)
    process=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert process.returncode==2 and b'Cannot write' in process.stderr
def test_missing_evaluator_assets_are_errors(tmp_path):
    import shutil
    folder=tmp_path/'evaluation';folder.mkdir()
    shutil.copyfile(ROOT/'evaluation/judge.py',folder/'judge.py')
    shutil.copyfile(ROOT/'evaluation/metrics.json',folder/'metrics.json')
    output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(folder/'judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(output)],cwd=tmp_path,capture_output=True)
    assert result.returncode==2
    report=json.loads(output.read_text())
    assert report['status']=='error' and report['metrics']=={} and report['errors']
