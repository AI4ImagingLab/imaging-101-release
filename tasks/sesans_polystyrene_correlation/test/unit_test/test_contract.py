from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def ref():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def test_reference_metric_zero(tmp_path):
    p=tmp_path/'p.npz';np.savez(p,**ref());r,c=j.evaluate(p)
    assert c==0 and r['passed'] and r['metrics']['profile_nrmse']==0 and r['metrics']['parameter_error']==0

@pytest.mark.parametrize('case',['missing_key','shape','dtype','nan','inf','complex','poor_profile','poor_parameters'])
def test_judge_rejection(case,tmp_path):
    x=ref()
    if case=='missing_key':del x['parameters']
    elif case=='shape':x['reconstruction']=x['reconstruction'][0]
    elif case=='dtype':x['reconstruction']=x['reconstruction'].astype('float32')
    elif case=='nan':x['reconstruction'][0,0]=np.nan
    elif case=='inf':x['parameters'][0,0]=np.inf
    elif case=='complex':x['reconstruction']=x['reconstruction'].astype(complex)
    elif case=='poor_profile':x['reconstruction']*=0
    elif case=='poor_parameters':x['parameters'][0,0]=.5
    p=tmp_path/'p.npz';np.savez(p,**x);r,c=j.evaluate(p)
    assert c==(1 if case.startswith('poor_') else 2)
    assert not r['passed']
    if c==2:assert r['metrics']=={} and r['errors']

def test_missing_file_and_evaluator_asset(tmp_path,monkeypatch):
    assert j.evaluate(tmp_path/'no.npz')[1]==2
    p=tmp_path/'p.npz';np.savez(p,**ref());monkeypatch.setattr(j,'HERE',tmp_path)
    assert j.evaluate(p)[1]==2

def test_cli_arbitrary_workdir_and_report_failure(tmp_path):
    script=ROOT/'evaluation/judge.py';p=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/r.json'
    good=subprocess.run([sys.executable,str(script),'--prediction',str(p),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert good.returncode==0 and json.loads(report.read_text())['passed']
    fail=subprocess.run([sys.executable,str(script),'--prediction',str(p),'--report',str(p/'impossible.json')],cwd=tmp_path,capture_output=True)
    assert fail.returncode==2 and b'report write failed' in fail.stderr

def test_profile_metric_known_shift(tmp_path):
    x=ref();x['reconstruction']+=.01;expected=.01/np.ptp(ref()['reconstruction'])
    p=tmp_path/'p.npz';np.savez(p,**x);r,c=j.evaluate(p)
    np.testing.assert_allclose(r['metrics']['profile_nrmse'],expected,rtol=1e-12)

def test_main_import_does_not_change_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before



def test_truth_precedence_and_malformed_hidden_asset(tmp_path,monkeypatch):
    import shutil
    hidden=tmp_path/'evaluation';hidden.mkdir()
    (tmp_path/'data').mkdir()
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    shutil.copy(ROOT/'evaluation/metrics.json',hidden/'metrics.json')
    baseline=ref();np.savez(hidden/'baseline_reference.npz',**baseline)
    truth={k:v.copy() for k,v in baseline.items()};truth['reconstruction']+=.01
    np.savez(hidden/'ground_truth.npz',**truth)
    prediction=tmp_path/'submission.npz';np.savez(prediction,**truth)
    monkeypatch.setattr(j,'HERE',hidden)
    report,code=j.evaluate(prediction)
    assert code==0 and report['metrics']['profile_nrmse']==0
    # An existing invalid truth is an evaluator error, never silently replaced
    # by an otherwise valid standard-method baseline.
    bad={k:v.copy() for k,v in truth.items()};bad['reconstruction']=bad['reconstruction'].astype(np.float32)
    np.savez(hidden/'ground_truth.npz',**bad)
    report,code=j.evaluate(prediction)
    assert code==2 and report['metrics']=={} and report['errors']

def test_public_metadata_and_clean_imports():
    import ast,importlib.util
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for value in ('1.41','2.70','0.0168','0.2','1.0003','501','length_standard_error'):
        assert value in text
    for domain in ('github.com','githubusercontent.com','github.io','git.io'):assert domain not in text.lower()
    for name in ('sasmodels','sasdata','bumps'):assert importlib.util.find_spec(name) is None
    allowed={'numpy','scipy','matplotlib','pathlib','json','argparse','sys','time','preprocessing','physics_model','solvers'}
    for p in list((ROOT/'src').glob('*.py'))+[ROOT/'evaluation/judge.py']:
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Import):assert all(n.name.split('.')[0] in allowed for n in node.names)
            elif isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] in allowed
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as a:
        assert set(a.files)=={'spin_echo_length','depolarisation','standard_error','wavelength','length_standard_error'}
