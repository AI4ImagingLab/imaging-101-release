from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np,pytest

@pytest.fixture
def judge(task_root):
    spec=importlib.util.spec_from_file_location('judge_test_module',task_root/'evaluation/judge.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def valid(task_root):
    with np.load(task_root/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def test_pass(task_root,judge):
    report,code=judge.evaluate(task_root/'output/reconstruction.npz')
    assert code==0 and report['passed'] and report['status']=='pass' and len(report['metrics'])==4

def test_poor(task_root,judge,tmp_path):
    a=valid(task_root);a['reconstruction'][:]=20;a['transmission'][:]=.5;p=tmp_path/'poor.npz';np.savez(p,**a)
    report,code=judge.evaluate(p);assert code==1 and report['status']=='fail' and not report['passed']

def test_zero_display_is_valid_poor(task_root,judge,tmp_path):
    a=valid(task_root);a['reconstruction'][:]=0;p=tmp_path/'zero.npz';np.savez(p,**a)
    report,code=judge.evaluate(p);assert code==1 and report['metrics']['reconstruction_ncc']==0

@pytest.mark.parametrize('case',['missing_file','missing_key','shape','dtype','nan','inf','complex','zero_transmission'])
def test_invalid(task_root,judge,tmp_path,case):
    p=tmp_path/'bad.npz';a=valid(task_root)
    if case=='missing_key':del a['transmission']
    elif case=='shape':a['transmission']=a['transmission'][0]
    elif case=='dtype':a['transmission']=a['transmission'].astype('float32')
    elif case=='complex':a['transmission']=a['transmission'].astype('complex128')
    elif case in ('nan','inf'):a['transmission'][0,0,0]=float(case)
    elif case=='zero_transmission':a['transmission'][:]=0
    if case!='missing_file':np.savez(p,**a)
    report,code=judge.evaluate(p)
    assert code==2 and report['metrics']=={} and report['status']=='error' and report['errors']

def test_evaluator_missing_assets(judge,tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    report,code=judge.evaluate(tmp_path/'none.npz');assert code==2 and report['metrics']=={} and report['errors']

def test_metric_configuration(judge,task_root,tmp_path,monkeypatch):
    config=json.loads((task_root/'evaluation/metrics.json').read_text());config['transmission_nrmse_boundary']=float('nan')
    (tmp_path/'metrics.json').write_text(json.dumps(config));monkeypatch.setattr(judge,'HERE',tmp_path)
    report,code=judge.evaluate(task_root/'output/reconstruction.npz');assert code==2 and report['metrics']=={}

def test_foreign_cwd_and_report_write(task_root,tmp_path):
    report=tmp_path/'nested'/'report.json'
    command=[sys.executable,str(task_root/'evaluation/judge.py'),'--prediction',str(task_root/'output/reconstruction.npz'),'--report',str(report)]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads(report.read_text())['status']=='pass'
    command[-1]=str(tmp_path)
    p=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr
