import importlib.util,subprocess,sys,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics_and_permutation():
    r={'positions':np.arange(225.).reshape(75,3),'moments':np.ones((75,3))}
    p={k:v[::-1].copy() for k,v in r.items()}
    assert judge.score(p,r)=={'position_rmse_um':0.,'moment_relative_l2':0.}
    p['moments']*=2
    assert judge.score(p,r)['moment_relative_l2']==1.

@pytest.mark.parametrize('fault',['missing','keys','shape','dtype','complex','nan','inf','poor'])
def test_bad_submissions(tmp_path,fault):
    f=tmp_path/'bad.npz'
    with np.load(ROOT/'evaluation/baseline_reference.npz') as d:r={k:d[k].copy() for k in d.files}
    expected=2
    if fault=='keys':r.pop('moments')
    elif fault=='shape':r['positions']=r['positions'][:,:1]
    elif fault=='dtype':r['positions']=r['positions'].astype(np.float32)
    elif fault=='complex':r['moments']=r['moments'].astype(complex)
    elif fault in ('nan','inf'):r['positions'][0,0,0]=float(fault)
    elif fault=='poor':r['moments']*=0;expected=1
    if fault!='missing':np.savez(f,**r)
    report,code=judge.evaluate(f)
    assert code==expected and not report['passed']
    if code==2:assert report['metrics']=={} and report['errors']

def test_missing_assets(tmp_path):
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={}

def test_cli_outside_working_directory_and_report_failure(tmp_path):
    args=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['status']=='pass'
    args[-1]=str(tmp_path)
    assert subprocess.run(args,cwd=tmp_path,capture_output=True).returncode==2

def test_full_cli(tmp_path):
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True)
    assert run.returncode==0,run.stderr.decode()
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz')
    assert code==0,report

def test_backend_unchanged():
    import matplotlib
    backend=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==backend

def test_direction_sign_and_axes():
    from src.visualization import angles
    inc,dec=angles(np.array([[0.,1.,0.],[1.,0.,0.],[0.,1.,-1.]]))
    np.testing.assert_allclose(inc,[0.,0.,45.],atol=1e-14)
    np.testing.assert_allclose(dec,[0.,90.,0.],atol=1e-14)
