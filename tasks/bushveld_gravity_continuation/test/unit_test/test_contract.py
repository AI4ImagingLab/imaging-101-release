import importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations
from src.physics_model import greens_matrix
from src.solvers import solve_coefficients
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def write(path,array=None,**keys):
    if array is not None:keys['reconstruction']=array
    np.savez_compressed(path,**keys);return path

def test_independent_metric_values():
    r=judge.score(np.array([2.,2.]),np.array([1.,2.]))
    assert r['ncc']==pytest.approx(6/np.sqrt(40))
    assert r['nrmse']==pytest.approx(1/np.sqrt(2))

@pytest.mark.parametrize('value',[0.,1000.])
def test_poor_output(tmp_path,value):
    r,c=judge.evaluate(write(tmp_path/'x.npz',np.full((1,48,64),value)))
    assert c==1 and r['status']=='fail' and not r['passed']

@pytest.mark.parametrize('kind',['shape','int','complex','nan','inf','key','missing','object'])
def test_invalid_submission(tmp_path,kind):
    x=np.ones((1,48,64));path=tmp_path/'x.npz'
    if kind=='shape':x=x[0]
    if kind=='int':x=x.astype(np.int64)
    if kind=='complex':x=x.astype(complex)
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='object':x=np.array([{'bad':1}],dtype=object)
    if kind=='key':write(path,other=x)
    elif kind!='missing':write(path,x)
    r,c=judge.evaluate(path)
    assert c==2 and r['metrics']=={} and r['errors']
    json.dumps(r,allow_nan=False)

def test_evaluator_missing_assets(tmp_path):
    r,c=judge.evaluate(write(tmp_path/'x.npz',np.ones((1,48,64))),tmp_path)
    assert c==2 and r['metrics']=={}

def test_invalid_boundary(tmp_path):
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":NaN,"nrmse_boundary":1}')
    r,c=judge.evaluate(tmp_path/'x.npz',tmp_path);assert c==2 and r['metrics']=={}

def test_truth_priority_and_invalid_reference(tmp_path):
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":0.9,"nrmse_boundary":0.1}')
    x=np.arange(48*64,dtype=float).reshape(1,48,64)
    write(tmp_path/'baseline_reference.npz',x);prediction=write(tmp_path/'prediction.npz',x)
    assert judge.evaluate(prediction,tmp_path)[1]==0
    write(tmp_path/'ground_truth.npz',x*2);assert judge.evaluate(prediction,tmp_path)[1]==1
    write(tmp_path/'ground_truth.npz',x[0]);assert judge.evaluate(prediction,tmp_path)[1]==2

def test_foreign_cwd_and_report(tmp_path):
    report=tmp_path/'new/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'notebooks/reference_outputs/upstream.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert json.loads(report.read_text())['passed']

def test_report_write_error(tmp_path):
    dest=tmp_path/'directory';dest.mkdir()
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'notebooks/reference_outputs/upstream.npz'),'--report',str(dest)],capture_output=True,text=True)
    assert r.returncode==2 and 'Report-write failure' in r.stderr

def test_standard_entrypoint_and_geometry():
    r=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert judge.evaluate(ROOT/'output/reconstruction.npz')[1]==0
    s,d,g=load_observations(ROOT/'data/raw_data.npz')
    assert np.all(g[:,:,2]==3000) and s[:,2].max()<3000
    np.testing.assert_allclose(g[0,:,0],np.linspace(s[:,0].min()+10000,s[:,0].max()-10000,64))
    np.testing.assert_allclose(g[:,0,1],np.linspace(s[:,1].min()+10000,s[:,1].max()-10000,48))
    text=(ROOT/'README.md').read_text()
    for value in ['306','3000','32735','600000','7200000','48','64','10000']:assert value in text
    assert 'github' not in text.lower()

def test_degenerate_solver_inputs():
    with pytest.raises(ValueError):greens_matrix(np.zeros((1,3)),np.zeros((1,3)))
    with pytest.raises(ValueError):solve_coefficients(np.ones((2,2)),np.ones(2))
    with pytest.raises(ValueError):solve_coefficients(np.eye(2),np.ones(2),damping=0)

def test_float32_output_is_scored_in_float64(tmp_path):
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as f:
        x=f['reconstruction'].astype(np.float32)
    assert judge.evaluate(write(tmp_path/'single.npz',x))[1]==0

@pytest.mark.parametrize('value',[0.,1.])
def test_constant_reference_is_evaluator_error(tmp_path,value):
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":0.9,"nrmse_boundary":0.1}')
    x=np.full((1,48,64),value)
    write(tmp_path/'baseline_reference.npz',x)
    report,code=judge.evaluate(write(tmp_path/'x.npz',x),tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_main_import_preserves_matplotlib_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
