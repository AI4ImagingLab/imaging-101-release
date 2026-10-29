from pathlib import Path
import importlib.util,subprocess,sys,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('task_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return {k:f[k].copy() for k in f.files}
def test_known_metrics():
    a=np.array([0.,1.,2.]);b=np.array([0.,2.,4.])
    n,e=judge.score(a,b)
    assert n==pytest.approx(1.);assert e==pytest.approx(np.sqrt(5/3)/4)
def test_zero_metric_rejected():
    with pytest.raises(ValueError):judge.score(np.zeros(3),np.zeros(3))
@pytest.mark.parametrize('problem',['missing','key','shape','dtype','nan','inf','complex','poor','zero'])
def test_bad_inputs(tmp_path,reference,problem):
    path=tmp_path/'submission.npz';d={k:v.copy() for k,v in reference.items()};k=next(iter(d))
    if problem=='key':del d[k]
    elif problem=='shape':d[k]=d[k][...,0]
    elif problem=='dtype':d[k]=d[k].astype(np.float32)
    elif problem=='complex':d[k]=d[k].astype(np.complex128)
    elif problem in ['nan','inf']:d[k].flat[0]=float(problem)
    elif problem=='poor':d={k:-v for k,v in d.items()}
    elif problem=='zero':d={k:np.zeros_like(v) for k,v in d.items()}
    if problem!='missing':np.savez_compressed(path,**d)
    report,code=judge.evaluate(path)
    assert code==(1 if problem in ['poor','zero'] else 2)
    assert report['status']==('fail' if problem in ['poor','zero'] else 'error')
    if code==2:assert report['metrics']=={} and report['errors']
def test_evaluator_assets_missing(tmp_path,reference):
    path=tmp_path/'pred.npz';np.savez(path,**reference)
    report,code=judge.evaluate(path,asset_dir=tmp_path)
    assert code==2 and report['metrics']=={}
def test_cli_any_cwd_and_nested_report(tmp_path,reference):
    path=tmp_path/'valid.npz';np.savez(path,**reference);report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert p.returncode==0,p.stderr;assert json.loads(report.read_text())['passed']
def test_report_write_failure(tmp_path,reference):
    path=tmp_path/'valid.npz';np.savez(path,**reference)
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Cannot write report' in p.stderr
def test_entrypoint():
    p=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True)
    assert p.returncode==0,p.stderr
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==0,report
def test_public_no_domain_library_or_urls():
    forbidden=['github.com','github.io','githubusercontent.com','git.io','codeload.']
    for f in [ROOT/'README.md',ROOT/'requirements.txt']:
        text=f.read_text(encoding='utf-8').lower();assert not any(x in text for x in forbidden)
    for f in (ROOT/'src').glob('*.py'):
        text=f.read_text();assert 'matplotlib.use' not in text
        assert 'import phasorpy' not in text and 'import dipy' not in text and 'import qmrpy' not in text
