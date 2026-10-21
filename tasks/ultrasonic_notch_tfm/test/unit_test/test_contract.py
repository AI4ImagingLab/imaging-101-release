from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def test_known_metrics():
    r=np.array([0.,1.,2.]); assert j.agreement(r,r)==pytest.approx((1,0))
    assert j.agreement(np.zeros(3),r)==pytest.approx((0,np.sqrt(5/3)/2))
    from src.visualization import agreement
    assert agreement(r+1,r)==pytest.approx(j.agreement(r+1,r))

@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf','poor','archive'])
def test_bad_prediction(case,tmp_path):
    path=tmp_path/'prediction.npz';r=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
    if case=='missing': pass
    elif case=='archive': path.write_text('not a zip')
    elif case=='key': np.savez(path,wrong=r)
    else:
        x=r.copy()
        if case=='shape':x=x[0]
        if case=='dtype':x=x.astype('float32')
        if case=='nan':x.flat[0]=np.nan
        if case=='inf':x.flat[0]=np.inf
        if case=='poor':x[:]=0
        np.savez(path,reconstruction=x)
    report,code=j.evaluate(path)
    assert code==(1 if case=='poor' else 2)
    assert not report['passed']
    if code==2: assert report['metrics']=={} and report['errors']

def test_truth_priority_and_asset_error(tmp_path):
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    report,code=j.evaluate(ROOT/'output/reconstruction.npz',tmp_path); assert code==2
    r=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];np.savez(tmp_path/'baseline_reference.npz',reconstruction=r)
    np.savez(tmp_path/'ground_truth.npz',reconstruction=r*2)
    assert j.evaluate(ROOT/'output/reconstruction.npz',tmp_path)[1]==1

def test_cli_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    cmd[-1]=str(tmp_path); run=subprocess.run(cmd,capture_output=True);assert run.returncode==2 and b'Cannot write' in run.stderr
