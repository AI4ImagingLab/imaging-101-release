from pathlib import Path
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metric_and_phase():
    u=np.eye(2,dtype=complex)[None]
    assert judge.score(1j*u,u)['phase_aligned_process_rmse']<1e-15
    assert judge.score(2*u,u)['unitarity_error']==pytest.approx(3*np.sqrt(2))
    assert judge.score(np.zeros_like(u),u)['phase_aligned_process_rmse']==pytest.approx(1/np.sqrt(2))

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','nan','inf','poor','phase'])
def test_submission_contract(tmp_path,kind):
    p=tmp_path/'submission.npz';ref=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
    if kind=='key':np.savez(p,other=ref)
    elif kind!='missing':
        x=ref.copy()
        if kind=='shape':x=x[0]
        if kind=='dtype':x=x.real
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        if kind=='poor':x=np.eye(2,dtype=complex)[None]
        if kind=='phase':x*=np.exp(.7j)
        np.savez(p,reconstruction=x)
    report,code=judge.evaluate(p)
    assert code==({'poor':1,'phase':0}.get(kind,2))
    if code==2:assert report['metrics']=={} and report['errors']

def test_evaluator_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(judge,'__file__',str(tmp_path/'judge.py'))
    r,c=judge.evaluate('unused.npz');assert c==2 and r['errors']

def test_visualization_metric_identity():
    from src.visualization import process_metrics
    u=np.eye(2,dtype=complex)[None]
    for x in [u,1j*u,2*u,np.zeros_like(u)]:
        assert process_metrics(x,u)==pytest.approx(judge.score(x,u))

def test_report_write_failure(tmp_path):
    import subprocess,sys
    blocked=tmp_path/'existing_file';blocked.write_text('not a directory')
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(blocked/'report.json')],capture_output=True)
    assert p.returncode==2 and b'Cannot write report' in p.stderr

def test_truth_precedence_and_baseline_fallback(tmp_path,monkeypatch):
    import shutil
    (tmp_path.parent/'data').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path.parent/'data/raw_data.npz')
    shutil.copyfile(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    prediction=tmp_path/'prediction.npz'
    np.savez(prediction,reconstruction=np.eye(2,dtype=complex)[None])
    monkeypatch.setattr(judge,'__file__',str(tmp_path/'judge.py'))
    assert judge.evaluate(prediction)[1]==1
    np.savez(tmp_path/'ground_truth.npz',reconstruction=np.eye(2,dtype=complex)[None])
    assert judge.evaluate(prediction)[1]==0
    # A malformed higher-priority asset must error, never fall back silently.
    np.savez(tmp_path/'ground_truth.npz',wrong=np.eye(2))
    assert judge.evaluate(prediction)[1]==2


def test_exact_identity_observation_and_global_phase():
    raw=dict(polar_angle=np.zeros((1,50)),azimuth=np.zeros((1,50)),quadrature_weight=np.full((1,50),1/50),expectation=np.zeros((1,2,4,50),complex))
    raw['expectation'][0,0,0]=1
    identity=np.eye(2,dtype=complex)[None]
    assert judge.observation_loss(identity,raw)==pytest.approx(0,abs=1e-15)
    assert judge.observation_loss(identity*np.exp(.713j),raw)==pytest.approx(0,abs=1e-15)
    wrong=np.array([[0,1j],[1j,0]])[None]
    assert judge.observation_loss(wrong,raw)>.1


def test_independent_better_expectation_fit_passes(tmp_path):
    from scipy.optimize import minimize
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:u=z['reconstruction'][0]
    u=u*np.exp(-.5j*np.angle(np.linalg.det(u)))
    initial=np.array([u[0,1].imag,u[0,1].real,u[0,0].imag,u[0,0].real])
    def matrix(q):
        q=q/np.linalg.norm(q);a,b,c,d=q
        return np.array([[d+1j*c,b+1j*a],[-b+1j*a,d-1j*c]])[None]
    before=judge.observation_loss(matrix(initial),raw)
    fit=minimize(lambda q:judge.observation_loss(matrix(q),raw),initial,method='BFGS',options={'gtol':1e-10})
    assert fit.fun<before-1e-8
    path=tmp_path/'improved.npz';np.savez(path,reconstruction=matrix(fit.x))
    report,code=judge.evaluate(path);assert code==0,report
