import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from src.physics_model import calibrated_phasors,mixture_phasors,OMEGA,FREQUENCY,REFERENCE_LIFETIME
from src.solvers import graphical_search,single_harmonic
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def good():
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def test_fourier_convention_and_calibration():
    # Explicit trigonometric moments independently check FFT sign, normalization,
    # standard intensity pooling and multi-harmonic correction.
    s=np.arange(1,98,dtype=np.uint8)[None,None,:]
    c=np.tile(np.arange(97,0,-1,dtype=np.uint8),(2,3,1))
    mean,z=calibrated_phasors(s,c)
    phase=2*np.pi*np.arange(97)/97
    for h in [1,2]:
        raw=np.sum(s*np.exp(1j*h*phase))/s.sum();ref=np.sum(c*np.exp(1j*h*phase))/c.sum()
        np.testing.assert_allclose(z[h-1,0,0],raw*(1/(1-1j*h*OMEGA*3.05))/ref,atol=1e-14)
    np.testing.assert_allclose(mean,s.mean(-1))

def test_exact_mixture_multiple_lifetimes():
    tau=np.array([[.2,.35,.6],[2.,3.,4.]])
    f=np.array([[.7,.4,.5],[.3,.6,.5]])
    z=mixture_phasors(tau,f);t,frac=graphical_search(z)
    np.testing.assert_allclose(t,tau,atol=1e-8);np.testing.assert_allclose(frac,f,atol=1e-8)
    known=.4;z1=mixture_phasors(np.array([[known],[3.4]]),np.array([[.5],[.5]]))[0]
    fraction,bound=single_harmonic(z1,np.ones_like(z1.real));np.testing.assert_allclose(bound,3.4);np.testing.assert_allclose(fraction,.5)

def test_known_metric_and_readme_constants():
    assert judge.normalized_error(np.array([1.,3.]),np.array([0.,2.]))==.5
    with pytest.raises(ValueError):judge.normalized_error(np.ones(2),np.ones(2))
    readme=(T/'README.md').read_text(encoding='utf8')
    assert '| Repetition frequency | 80 |' in readme and '| Standard lifetime | 3.05 |' in readme
    assert FREQUENCY==80 and REFERENCE_LIFETIME==3.05

def test_review_forward_metric_known_case():
    from src.visualization import forward_residual
    tau=np.array([[[.4]],[[3.4]]]);fraction=np.array([[.4]])
    output={'reconstruction':tau[None],'bound_fraction':fraction[None]}
    z=mixture_phasors(tau,np.stack([1-fraction,fraction]))
    np.testing.assert_allclose(forward_residual(output,z,np.ones((1,1),bool)),0,atol=1e-15)
    np.testing.assert_allclose(forward_residual(output,z+.1+.2j,np.ones((1,1),bool)),np.sqrt(.05),atol=1e-15)

@pytest.mark.parametrize('kind',['missingkey','shape','integer','complex','nan','inf','unordered','fraction','background'])
def test_bad_submission_errors(tmp_path,good,kind):
    if kind=='missingkey':del good['bound_fraction']
    elif kind=='shape':good['reconstruction']=good['reconstruction'][0]
    elif kind=='integer':good['reconstruction']=good['reconstruction'].astype(np.int32)
    elif kind=='complex':good['reconstruction']=good['reconstruction'].astype(np.complex128)
    elif kind in ['nan','inf']:good['reconstruction'][0,0,0,0]=float(kind)
    elif kind=='unordered':good['reconstruction']=good['reconstruction'][:,::-1].copy()
    elif kind=='fraction':good['bound_fraction'][0,0,0]=2
    else:good['reconstruction'][0,0,0,0]=1
    p=tmp_path/'bad.npz';np.savez(p,**good);report,code=judge.evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_poor_valid_submission_fails(tmp_path,good):
    good['reconstruction']*=1.5;good['bound_fraction']*=.5
    p=tmp_path/'poor.npz';np.savez(p,**good);r,c=judge.evaluate(p)
    assert c==1 and r['status']=='fail' and not r['passed']

def test_missing_and_hidden_errors(tmp_path):
    r,c=judge.evaluate(tmp_path/'missing.npz');assert c==2 and r['metrics']=={}
    r,c=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path);assert c==2 and r['metrics']=={}

def test_cli_from_other_directory(tmp_path):
    report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads(report.read_text())['passed']
    blocked=tmp_path/'file';blocked.write_text('file')
    p=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Report write failed' in p.stderr

def test_main_import_no_backend_change():
    import matplotlib
    before=matplotlib.get_backend()
    from src import main
    assert matplotlib.get_backend()==before

def test_true_reference_precedence(tmp_path,good):
    # Hidden actual truth is not supplied in this task; test contractual priority
    # with a deliberately malformed hypothetical truth and a valid baseline.
    np.savez(tmp_path/'baseline_reference.npz',**good)
    np.savez(tmp_path/'ground_truth.npz',wrong=np.ones(2))
    r,c=judge.evaluate(T/'evaluation/baseline_reference.npz',tmp_path)
    assert c==2 and 'Missing required key' in r['errors'][0]
