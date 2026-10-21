import importlib.util
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import initialize,prepare_psf,load_data
from src.physics_model import make_interpolator,model,residual
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_initialization_analytic():
    a=np.full((17,17),3.);a[8,9]=13
    np.testing.assert_array_equal(initialize(a),[10,9,8,3])

def test_physical_flux_linearity_translation_and_sky():
    _,flight=load_data(ROOT/'data/raw_data.npz');psf=prepare_psf(flight)
    assert psf.sum()==pytest.approx(16)
    f=make_interpolator(psf)
    p=np.array([100.,8,8,0])
    np.testing.assert_allclose(model(p* np.array([2,1,1,1]),f),2*model(p,f))
    q=p.copy();q[1]+=1
    np.testing.assert_allclose(model(q,f)[:,1:],model(p,f)[:,:-1],atol=1e-13)
    q=p.copy();q[3]=2
    np.testing.assert_allclose(model(q,f)-model(p,f),2)
    # Pixel sampled flux is approximate; oversampled sum is exactly calibrated.
    assert .95<model(p,f).sum()/p[0]<1.05

def test_exact_synthetic_identifiability():
    _,flight=load_data(ROOT/'data/raw_data.npz');f=make_interpolator(prepare_psf(flight))
    parameters=np.array([1000.,8.2,7.8,4.])
    image=model(parameters,f)
    result,_=reconstruct(np.stack([image]*4),flight)
    np.testing.assert_allclose(result['positions'][0],np.tile(parameters[1:3],(4,1)),atol=1e-6)
    np.testing.assert_allclose(result['flux'],1000,atol=1e-4)
    np.testing.assert_allclose(result['background'],4,atol=1e-6)

def test_residual_gradient_finite_difference():
    _,flight=load_data(ROOT/'data/raw_data.npz');f=make_interpolator(prepare_psf(flight))
    p=np.array([1000.,8.2,7.8,4]);image=model(p,f)+.1
    r=residual(p,f,image)
    steps=[1e-3,1e-5,1e-5,1e-5]
    numerical=[]
    for k,h in enumerate(steps):
        d=np.zeros(4);d[k]=h
        numerical.append((np.sum(residual(p+d,f,image)**2)-np.sum(residual(p-d,f,image)**2))/(2*h))
    y,x=np.indices(image.shape,dtype=float);xi=4*(x-p[1])+40;yi=4*(y-p[2])+40
    jac=np.stack([f(xi,yi,grid=False),-4*p[0]*f(xi,yi,dx=1,grid=False),-4*p[0]*f(xi,yi,dy=1,grid=False),np.ones_like(x)],axis=-1).reshape(-1,4)
    np.testing.assert_allclose(numerical,2*jac.T@r,rtol=1e-5,atol=1e-5)

@pytest.fixture
def good():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf','complex','negative_flux'])
def test_judge_malformed(tmp_path,good,case):
    path=tmp_path/'prediction.npz'
    if case=='key':del good['flux']
    if case=='shape':good['flux']=good['flux'].ravel()
    if case=='dtype':good['flux']=good['flux'].astype('float32')
    if case in ('nan','inf'):good['flux'][0,0]=float(case)
    if case=='complex':good['flux']=good['flux'].astype(complex)
    if case=='negative_flux':good['flux'][0,0]=-1
    if case!='missing':np.savez(path,**good)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_judge_good_poor_and_missing_assets(tmp_path,good):
    path=tmp_path/'p.npz';np.savez(path,**good)
    report,code=judge.evaluate(path);assert code==0 and report['passed']
    good['flux']*=.5;np.savez(path,**good)
    report,code=judge.evaluate(path);assert code==1 and report['status']=='fail'
    report,code=judge.evaluate(path,tmp_path);assert code==2 and not report['metrics']

def test_metadata_agreement():
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for value in ('0.30325','1.2','0.3','17','81','4.5'): assert value in readme

@pytest.mark.parametrize('case',['corrupt','object','invalid_boundary'])
def test_invalid_archives_and_configuration(tmp_path,good,case):
    path=tmp_path/'prediction.npz'
    assets=ROOT/'evaluation'
    if case=='corrupt':path.write_bytes(b'broken archive')
    elif case=='object':
        good['flux']=good['flux'].astype(object);np.savez(path,**good)
    else:
        import json,shutil
        np.savez(path,**good);assets=tmp_path/'assets';assets.mkdir()
        shutil.copy(ROOT/'evaluation/baseline_reference.npz',assets)
        config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'))
        config['position_error_boundary']=-1
        (assets/'metrics.json').write_text(json.dumps(config),encoding='utf-8')
    report,code=judge.evaluate(path,assets)
    assert code==2 and report['metrics']=={} and report['errors']

def test_cli_report_write_failure(tmp_path,good):
    import os,sys,subprocess
    path=tmp_path/'valid.npz';np.savez(path,**good)
    destination=tmp_path/'directory';destination.mkdir()
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(destination)],cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env=env)
    assert run.returncode==2 and 'Report write failed' in run.stderr

def test_metric_definitions_known_case(tmp_path,good):
    good['positions'][0,0]+=[3e-6,4e-6]
    good['flux']*=.5
    good['background']+=3
    path=tmp_path/'known.npz';np.savez(path,**good)
    report,code=judge.evaluate(path)
    assert code==1
    assert report['metrics']['position_error']==pytest.approx(5e-6)
    assert report['metrics']['flux_relative_error']==pytest.approx(.5)
    assert report['metrics']['background_absolute_error']==pytest.approx(3)
