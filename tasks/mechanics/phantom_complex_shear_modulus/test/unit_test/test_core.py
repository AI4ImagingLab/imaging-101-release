from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.preprocessing import load_data
from src.physics_model import polynomial_kernels,polynomial_filter,planar_laplacian,helmholtz_inversion,relative_residual
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('mre_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_raw_contract_and_readme():
    wave,mask,f,c=load_data();assert wave.shape==(2,128,80,25,3) and mask.sum()==144637
    assert np.array_equal(f,[80.,100.])
    with np.load('data/raw_data.npz') as z:
        readme=(ROOT/'README.md').read_text(encoding='utf-8')
        for k in z.files:assert '| '+k+' |' in readme
    for a in c:assert np.allclose(np.diff(a),.0015,rtol=0,atol=3e-17)
    assert 'github' not in readme.lower()
    for command in ['mathbf','omega','Delta','rho','simeq','qquad','pi','frac','arccos']:assert chr(92)+command in readme
    assert 'reflected 9-by-9' in readme and '0.00000001' in readme

def test_polynomial_laplacian_units_and_no_z_derivative():
    axis=np.arange(30)*.0015;x,y,z=np.meshgrid(axis,axis,np.arange(3)*.0015,indexing='ij')
    field=(2*x*x+3*y*y+7*z*z)[...,None].astype(complex)
    lap=planar_laplacian(field,axis,axis)
    assert np.allclose(lap[5:-5,5:-5],10.,atol=5e-12)
    # z-only curvature is intentionally absent from the selected planar operator.
    field=(7*z*z)[...,None].astype(complex)
    assert np.max(abs(planar_laplacian(field,axis,axis)))<1e-12

def test_analytic_complex_helmholtz_and_scale():
    # Independently known local vector Helmholtz pair, with SI density/frequency.
    rng=np.random.default_rng(4);u=(rng.normal(size=(7,8,2,3))+1j*rng.normal(size=(7,8,2,3)))*.01
    f=80.;known=10000+2000j;lu=-1000*(2*np.pi*f)**2*u/known
    result=helmholtz_inversion(u,lu,f)
    assert np.allclose(result,known,rtol=1e-7,atol=1e-5)
    scaled=helmholtz_inversion(2*u*np.exp(.7j),2*lu*np.exp(.7j),f)
    assert np.allclose(scaled,result,rtol=1e-7,atol=1e-5)
    assert np.max(abs(helmholtz_inversion(u,lu,2*f)/result-4))<1e-14
    assert relative_residual(np.full(u.shape[:-1],known),u,lu,f,np.ones(u.shape[:-1],bool))<1e-15

def test_metric_known_case():
    a=np.array([1.,3.]);b=np.array([2.,4.]);r=judge.metrics(a,b)
    assert r['ncc']==pytest.approx(14/np.sqrt(200));assert r['nrmse']==.5
    with pytest.raises(ValueError):judge.metrics(a,np.ones(2))
    assert judge.metrics(np.zeros(2),b)['ncc']==0

@pytest.mark.parametrize('kind',['missing','key','extra','shape','dtype','real','nan','inf','object','broken','exterior'])
def test_invalid_contract(tmp_path,kind):
    p=tmp_path/'bad.npz';a=np.zeros(judge.SHAPE,np.complex128)
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=a)
    elif kind=='extra':np.savez(p,reconstruction=a,extra=[1])
    elif kind=='shape':np.savez(p,reconstruction=np.zeros((2,25,80,128),complex))
    elif kind=='dtype':np.savez(p,reconstruction=a.astype(np.complex64))
    elif kind=='real':np.savez(p,reconstruction=a.real)
    elif kind=='object':np.savez(p,reconstruction=np.array([{}],object))
    elif kind=='broken':p.write_bytes(b'broken')
    else:
        a[0,0,0,0,0]=np.nan if kind=='nan' else np.inf if kind=='inf' else 1e-12;np.savez(p,reconstruction=a)
    r=judge.evaluate(p);assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_poor_and_missing_evaluator_assets(tmp_path):
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=np.zeros(judge.SHAPE,complex));r=judge.evaluate(p);assert r['status']=='fail' and not r['passed']
    assert judge.evaluate(p,tmp_path)['status']=='error'
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":NaN}')
    assert judge.evaluate(p,tmp_path)['status']=='error'

def test_both_complex_components_scored(tmp_path):
    a=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];a=a.real.astype(complex);p=tmp_path/'missing_loss.npz';np.savez(p,reconstruction=a)
    r=judge.evaluate(p);assert r['status']=='fail' and r['metrics']['loss_0_ncc']==0

def test_judge_cli_foreign_cwd_and_report_error(tmp_path):
    import subprocess,sys,json
    def call(p,r):return subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True)
    report=tmp_path/'nested/report.json';assert call(ROOT/'evaluation/baseline_reference.npz',report).returncode==0
    assert json.loads(report.read_text())['passed']
    assert call(tmp_path/'missing.npz',report).returncode==2
    poor=tmp_path/'poor.npz';np.savez(poor,reconstruction=np.zeros(judge.SHAPE,complex));assert call(poor,report).returncode==1
    blocker=tmp_path/'file';blocker.write_text('x');assert call(poor,blocker/'report.json').returncode==2

def test_main_import_no_backend_mutation():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert before==matplotlib.get_backend()
