import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data
from src.solvers import clip_parameters,pivoted_qr_step
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics():
    a=np.array([1.,2.,3.]);assert judge.metrics(a,a)==pytest.approx((1.,0.))
    assert judge.metrics(np.zeros(3),a)==pytest.approx((0.,np.sqrt(14/3)/2))

def test_azimuth_periodic_boundary_case():
    eps=1e-7
    agreement,error=judge.azimuth_metrics(np.array([eps,np.pi+eps]),np.array([np.pi-eps,-eps]))
    assert error==pytest.approx(2*eps/np.pi,rel=1e-8)
    assert agreement==pytest.approx(np.cos(4*eps),abs=1e-15)
    assert judge.azimuth_metrics(np.array([0.]),np.array([np.pi]))==pytest.approx((1.,0.),abs=1e-15)

def test_qr_solve_does_not_use_diagnostic_rank_threshold():
    a=np.diag([1.,1e-11]);b=np.array([1.,1e-11])
    np.testing.assert_allclose(pivoted_qr_step(a,b),[1,1],rtol=1e-14)

def test_clipping_and_original_azimuth_wrap():
    m=np.array([[-1.,4.,4.,20.,-1.,150.,20.,-1.,15.]])
    v=clip_parameters(m)[0]
    assert v[0]==0 and v[1]==np.pi and v[2]==4-float(np.float32(3.1415926))
    assert tuple(v[3:])==(13.,.001,120.,10.,0.,10.)

def test_public_data_and_readme_contract():
    d=load_data();assert d['raw'].shape==(64,96,4,30)
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert 'github' not in text.lower()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'stokes','wavelength','cavity_velocity','spectral_psf_6301','spectral_psf_6302'}
        for key in z.files: assert '| '+key+' |' in text
    assert all(s in text for s in ['0.93','0.059','6301.4995','6302.4931'])

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,True,'0.1'])
def test_invalid_error_boundary(value):
    with pytest.raises((ValueError,KeyError)):judge.finite_boundary({'b':value},'b')

def test_missing_and_invalid_ncc_boundaries():
    with pytest.raises(KeyError):judge.finite_boundary({},'b')
    for value in [-1.01,1.01]:
        with pytest.raises(ValueError):judge.finite_boundary({'b':value},'b',True)

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf','complex'])
def test_bad_submission(tmp_path,kind):
    p=tmp_path/'prediction.npz';report=tmp_path/'report.json'
    a=np.zeros((1,64,96,9))
    if kind=='key':np.savez(p,bad=a)
    elif kind=='shape':np.savez(p,reconstruction=a[:,:,:-1])
    elif kind=='dtype':np.savez(p,reconstruction=a.astype('f4'))
    elif kind=='complex':np.savez(p,reconstruction=a.astype('c16'))
    elif kind in ('nan','inf'):
        a.flat[0]=float(kind);np.savez(p,reconstruction=a)
    elif kind!='missing':np.savez(p,reconstruction=a)
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)],capture_output=True)
    result=json.loads(report.read_text());assert proc.returncode==2 and result['status']=='error' and result['metrics']=={}

def test_bad_physical_prediction_fails(tmp_path):
    # A finite physical atmosphere that is unrelated to these observed spectra.
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a=z['reconstruction'].copy()
    a[...,0]=0.;a[...,3]+=5.
    path=tmp_path/'bad.npz';np.savez(path,reconstruction=a)
    report=judge.evaluate(path)
    assert not report['passed'] and any(report['metrics'][c+'_whitened_rms']>report['thresholds'][c+'_whitened_rms']['boundary'] for c in 'IQUV')

def test_independent_observation_source_case():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a=z['reconstruction'][0]
    expected=np.asarray(json.loads((ROOT/'evaluation/metrics.json').read_text())['baseline'][0]['stokes_whitened_rms'])
    np.testing.assert_allclose(judge.observation_rms(a,ROOT/'data/raw_data.npz'),expected,rtol=3e-6,atol=2e-5)

def test_actual_observation_threshold_is_consumed(tmp_path,monkeypatch):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:a=z['reconstruction']
    np.savez(tmp_path/'prediction.npz',reconstruction=a)
    config=json.loads((ROOT/'evaluation/metrics.json').read_text());config['I_whitened_rms_boundary']=0.
    # Isolated evaluator directory; public calibration location remains identical.
    original=json.loads
    def altered(text,*args,**kwargs):return config if 'I_whitened_rms_boundary' in text else original(text,*args,**kwargs)
    monkeypatch.setattr(judge.json,'loads',altered)
    result=judge.evaluate(tmp_path/'prediction.npz')
    assert not result['passed'] and result['thresholds']['I_whitened_rms']['boundary']==0

def test_rejected_residual_requires_same_linearization_point():
    # f(x)=exp(x), target3, bestx0: a large Gauss-Newton trial is worse.
    # Reusing that rejected residual reverses the next step's sign, even though
    # the best model/Jacobian have not moved. This is the identified source bug.
    best=0.;target=3.;jac=np.exp(best);residual=target-np.exp(best)
    trial=best+jac*residual/(jac*jac*1.01)
    rejected=target-np.exp(trial)
    assert rejected**2>residual**2 and rejected<0<residual
    restored=target-np.exp(best)
    assert jac*restored>0 and jac*rejected<0

def test_evaluator_assets_and_nonfinite_arithmetic(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'HERE',tmp_path)
    with pytest.raises(FileNotFoundError):judge.evaluate(tmp_path/'missing.npz')
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError):judge.metrics(np.full(3,1e308),np.arange(3.))

def test_prior_domain_is_enforced_for_all_parameters(tmp_path):
    with np.load(ROOT/'output/reconstruction.npz') as z:a=z['reconstruction'].copy()
    a[...,4]=-.01
    path=tmp_path/'negative_width.npz';np.savez(path,reconstruction=a)
    result=judge.evaluate(path)
    assert not result['passed'] and result['metrics']['parameter_domain_violation']>0


def test_report_write_failure_and_foreign_cwd(tmp_path):
    blocked=tmp_path/'blocked';blocked.write_text('file')
    proc=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'absent.npz'),'--report',str(blocked/'report.json')],cwd=tmp_path,capture_output=True)
    assert proc.returncode==2 and b'report write failed' in proc.stderr
    good=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'foreign.json')],cwd=tmp_path,capture_output=True)
    assert good.returncode==0 and json.loads((tmp_path/'foreign.json').read_text())['passed']
