from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from src.physics_model import FourierGrid,SHGFrog,scaled_residual
from src.preprocessing import load_observations,gaussian_initialization
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',T/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_dense_fourier_parseval():
    g=FourierGrid();rng=np.random.RandomState(3);x=rng.normal(size=512)+1j*rng.normal(size=512)
    dense=g.dt/(2*np.pi)*np.exp(1j*g.w[:,None]*g.t)@x
    assert np.linalg.norm(g.forward(x)-dense)/np.linalg.norm(dense)<1e-10
    np.testing.assert_allclose(g.backward(g.forward(x)),x,rtol=1e-12,atol=1e-12)
    assert abs(np.sum(abs(x)**2)*g.dt/(2*np.pi*np.sum(abs(g.forward(x))**2)*g.dw)-1)<1e-12

def test_fractional_delay_and_absolute_shg():
    g=FourierGrid();tau=np.array([-37.3,0.,42.7])*1e-15;m=SHGFrog(g,tau);a=2*np.log(2)/(500e-15)**2;e=np.exp(-a*g.t**2);s,A,_,_=m.signal(g.forward(e))
    np.testing.assert_allclose(A,np.exp(-a*(g.t[None]-tau[:,None])**2),rtol=0,atol=1e-12)
    expected=np.exp(-a*tau[:,None]**2)*np.exp(-g.w[None]**2/(4*a))/(8*np.pi*a)
    assert np.linalg.norm(m.trace(g.forward(e))-expected)/np.linalg.norm(expected)<1e-9

def test_projection_and_complex_gradient():
    d,m=load_observations(T/'data/raw_data.npz');x=gaussian_initialization(m.grid);ids=np.array([240,255,269]);s,A,E,H=m.signal(x,ids);_,mu,_=scaled_residual(d,m.trace(x));p=m.project(d[ids]/mu,s)
    np.testing.assert_allclose(abs(m.grid.forward(p))**2,d[ids]/mu,rtol=1e-10,atol=1e-35)
    rng=np.random.RandomState(9);v=rng.normal(size=512)+1j*rng.normal(size=512);v*=np.linalg.norm(x)/np.linalg.norm(v);h=1e-5
    f=lambda z:np.sum(abs(p-m.signal(z,ids)[0])**2)
    fd=(f(x+h*v)-f(x-h*v))/(2*h);analytic=np.real(np.vdot(m.gradient(p,s,A,E,H).sum(axis=0),v))
    assert abs(fd-analytic)/abs(analytic)<1e-5

def test_ambiguity_alignment_and_scale():
    g=FourierGrid();e=np.exp(-(g.t/400e-15)**2)*(1+.2j*np.sin(g.t/200e-15));e/=np.sqrt(np.sum(abs(e)**2)*g.dt)
    a=g.forward(e);q=.137;shift=g.backward(a*np.exp(1j*np.arange(512)*q))*np.exp(.61j)
    assert judge.align_field(shift,e,g.dw)['complex_relative_l2']<1e-6
    reversal=np.conj(np.roll(e[::-1],1));assert judge.align_field(reversal,e,g.dw)['complex_relative_l2']<1e-6
    assert judge.align_field(2*e,e,g.dw)['complex_relative_l2']>.99

def test_public_constants_and_data_sufficiency():
    d,m=load_observations(T/'data/raw_data.npz');text=(T/'README.md').read_text(encoding='utf8')
    for v in ['407474458795.6243','903760270468365.0','3.0116847910804345e-14','3.0096e-14']:assert v in text
    assert not np.array_equal(m.delays,m.grid.t)
    with np.load(T/'data/raw_data.npz',allow_pickle=False) as f:assert set(f.files)=={'trace','delay','time','angular_frequency'}
    assert 'github' not in text.lower()

@pytest.mark.parametrize('case',['missing','key','shape','dtype','nan','inf'])
def test_judge_malformed(tmp_path,case):
    p=tmp_path/'prediction.npz';a=np.ones((1,512),complex)
    if case=='key':np.savez(p,wrong=a)
    elif case=='shape':np.savez(p,reconstruction=a[:,:3])
    elif case=='dtype':np.savez(p,reconstruction=a.astype(np.complex64))
    elif case in ['nan','inf']:
        a[0,3]=np.nan if case=='nan' else np.inf;np.savez(p,reconstruction=a)
    elif case!='missing':np.savez(p,reconstruction=a)
    r,c=judge.evaluate(p);assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']

def test_judge_valid_zero_and_amplitude_fail(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=np.zeros((1,512),np.complex128));r,c=judge.evaluate(p);assert c==1 and not r['passed'] and all(np.isfinite(list(r['metrics'].values())))
    with np.load(T/'evaluation/baseline_reference.npz') as f:e=f['reconstruction']
    np.savez(p,reconstruction=2*e);r,c=judge.evaluate(p);assert c==1 and r['metrics']['energy_relative_error']>2.9

def test_judge_foreign_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert r.returncode==0,r.stderr
    bad=tmp_path/'directory';bad.mkdir();r=subprocess.run(cmd[:-1]+[str(bad)],cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Cannot write report' in r.stderr
    r,c=judge.evaluate(T/'output/reconstruction.npz',assets=tmp_path);assert c==2 and r['metrics']=={} and 'Evaluator' in r['errors'][0]
