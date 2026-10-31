import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def test_metric_cases():
    assert j.score(np.array([0.,1.]),np.array([0.,1.]))==dict(ncc=1.,nrmse=0.)
    assert j.score(np.zeros(2),np.array([0.,1.]))['ncc']==0.
def test_pass_and_quality_fail(tmp_path):
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==0
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=np.zeros((1,2,4801)));report,code=j.evaluate(p);assert code==1 and report['status']=='fail'
def test_malformed(tmp_path):
    cases=[dict(other=np.zeros(1)),dict(reconstruction=np.zeros((2,4801))),dict(reconstruction=np.zeros((1,2,4801),dtype='int32')),dict(reconstruction=np.zeros((1,2,4801),dtype='complex128')),dict(reconstruction=np.full((1,2,4801),np.nan)),dict(reconstruction=np.full((1,2,4801),np.inf))]
    for i,case in enumerate(cases):
        p=tmp_path/f'bad{i}.npz';np.savez(p,**case);r,c=j.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
    assert j.evaluate(tmp_path/'missing.npz')[1]==2
def test_foreign_cwd_cli_report_and_report_failure(tmp_path):
    report=tmp_path/'nested/r.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(report)]
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==0;assert json.loads(report.read_text())['passed']
    cmd[-1]=str(tmp_path);p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Report write failure' in p.stderr
def test_missing_assets_and_bad_thresholds(tmp_path,monkeypatch):
    monkeypatch.setattr(j,'HERE',tmp_path)
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
    import shutil
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    (tmp_path/'metrics.json').write_text('{"ncc_boundary": null,"nrmse_boundary": 0.1}')
    assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==2
def test_truth_priority(tmp_path,monkeypatch):
    import shutil
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz');shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:np.savez(tmp_path/'ground_truth.npz',reconstruction=-f['reconstruction'])
    monkeypatch.setattr(j,'HERE',tmp_path);assert j.evaluate(ROOT/'evaluation/baseline_reference.npz')[1]==1
