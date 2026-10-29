import importlib.util,json,subprocess,sys
import numpy as np,pytest

@pytest.fixture
def judge(root):
    spec=importlib.util.spec_from_file_location('judge_test',root/'evaluation/judge.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def true_image(root):
    with np.load(root/'evaluation/ground_truth.npz',allow_pickle=False) as f:return f['reconstruction'].copy()

def test_fresh_and_true(root,judge):
    report,code=judge.evaluate(root/'output/reconstruction.npz');assert code==0 and report['passed'] and len(report['metrics'])==3
    report,code=judge.evaluate(root/'evaluation/ground_truth.npz');assert code==0

@pytest.mark.parametrize('mode',['zero','midpoint','wrong_intervals'])
def test_valid_poor(root,judge,tmp_path,mode):
    a=true_image(root)
    if mode=='zero':a[:]=0
    elif mode=='midpoint':
        with np.load(root/'data/raw_data.npz') as f:a=(f['lower']+f['upper'])/2
    else:a[:]=.5
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=a);r,c=judge.evaluate(p)
    assert c==1 and r['status']=='fail' and not r['passed'] and r['metrics']

@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','range'])
def test_invalid(root,judge,tmp_path,case):
    a=true_image(root);p=tmp_path/'bad.npz';k='reconstruction'
    if case=='key':k='other'
    elif case=='shape':a=a[0]
    elif case=='dtype':a=a.astype(np.float64)
    elif case=='complex':a=a.astype(np.complex64)
    elif case in ('nan','inf'):a[0,0,0,0]=float(case)
    elif case=='range':a[0,0,0,0]=2.
    if case!='missing':np.savez(p,**{k:a})
    report,code=judge.evaluate(p);assert code==2 and report['metrics']=={} and report['errors']

def test_missing_evaluator_and_bad_config(root,judge,tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    r,c=judge.evaluate(root/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
    config=json.loads((root/'evaluation/metrics.json').read_text());config['ncc_boundary']=float('inf');(tmp_path/'metrics.json').write_text(json.dumps(config))
    r,c=judge.evaluate(root/'output/reconstruction.npz');assert c==2 and r['metrics']=={}

def test_foreign_cwd_and_report_failure(root,tmp_path):
    report=tmp_path/'nested/report.json';cmd=[sys.executable,str(root/'evaluation/judge.py'),'--prediction',str(root/'output/reconstruction.npz'),'--report',str(report)]
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==0 and json.loads(report.read_text())['passed']
    cmd[-1]=str(tmp_path);r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report write failed' in r.stderr
