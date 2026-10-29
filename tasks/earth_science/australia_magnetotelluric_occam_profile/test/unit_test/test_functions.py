from pathlib import Path
import importlib.util
import numpy as np
import pytest
from src.preprocessing import load_data
from src.physics_model import response,jacobian,weighted_rms
from src.solvers import roughness_matrix,occam_update

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_homogeneous_halfspace():
    f=np.logspace(-5,5,20)
    rho,phase=response(f,np.array([2.]),np.array([0.]))
    np.testing.assert_allclose(rho,2.,atol=1e-14)
    np.testing.assert_allclose(phase,np.pi/4,atol=1e-14)

def test_layer_split_invariance():
    f=np.logspace(-3,3,20)
    a=response(f,np.array([1.,3.]),np.array([0.,500.]))
    b=response(f,np.array([1.,1.,3.]),np.array([0.,200.,500.]))
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-13)

def test_jacobian_central_differences():
    d=load_data(ROOT/'data/raw_data.npz');m=1.7+.5*np.sin(np.arange(25));f=1/d['period']
    j1,j2,_=jacobian(f,m,d['layer_top']);h=1e-5
    for j in range(25):
        a=m.copy();b=m.copy();a[j]+=h;b[j]-=h
        x=response(f,a,d['layer_top']);y=response(f,b,d['layer_top'])
        np.testing.assert_allclose(j1[:,j],(x[0]-y[0])/(2*h),rtol=2e-5,atol=2e-9)
        np.testing.assert_allclose(j2[:,j],(x[1]-y[1])/(2*h),rtol=2e-5,atol=2e-9)

def test_stable_large_attenuation():
    r,p=response(np.array([50.]),np.array([-1.,1.]),np.array([0.,100000.]))
    assert np.isfinite(r).all() and np.isfinite(p).all()
    np.testing.assert_allclose([r[0],p[0]],[-1.,np.pi/4],atol=1e-13)

def test_roughness_and_update_normal_equations():
    d=roughness_matrix(4);np.testing.assert_array_equal(d@np.ones(4),0)
    j1=np.eye(4);j2=np.eye(4)*.25;m=np.arange(4.)
    u=occam_update(m,np.ones(4),np.zeros(4),np.ones(4),np.ones(4),j1,j2,3.)
    np.testing.assert_allclose((3*d.T@d+1.0625*np.eye(4))@u,1+1.0625*m,atol=1e-13)

def test_independent_metrics():
    a=np.array([1.,2.,3.]);b=np.array([2.,2.,4.])
    n,e=judge.similarity(a,b)
    assert n==pytest.approx(18/np.sqrt(14*24))
    assert e==pytest.approx(np.sqrt(2/3)/2)
    assert weighted_rms([1,3],[0,1],[1,2])==pytest.approx(1.)

@pytest.mark.parametrize('kind',['missing_key','shape','float32','complex','nan','inf','overflow','underflow'])
def test_judge_invalid(tmp_path,kind):
    a=np.ones((1,25),dtype=float);key='reconstruction'
    if kind=='missing_key':key='other'
    if kind=='shape':a=a[0]
    if kind=='float32':a=a.astype('float32')
    if kind=='complex':a=a.astype(complex)
    if kind=='nan':a[0,0]=np.nan
    if kind=='inf':a[0,0]=np.inf
    if kind=='overflow':a[:]=1e308
    if kind=='underflow':a[:]=-1e308
    pred=tmp_path/'bad.npz';np.savez(pred,**{key:a})
    report,code=judge.evaluate(pred)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_judge_missing_and_bad_archive(tmp_path):
    assert judge.evaluate(tmp_path/'missing.npz')[1]==2
    p=tmp_path/'bad.npz';p.write_bytes(b'not an archive')
    assert judge.evaluate(p)[1]==2

def test_judge_fail_quality(tmp_path):
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=np.full((1,25),2.))
    report,code=judge.evaluate(p)
    assert code==1 and report['status']=='fail' and np.isfinite(list(report['metrics'].values())).all()

def test_evaluator_failure(tmp_path):
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['errors'] and report['metrics']=={}
