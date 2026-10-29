from pathlib import Path
import importlib.util,subprocess,sys,shutil,json
import numpy as np
import pytest
from src.physics_model import scattering_vector,correlation_forward,msd_from_field,modulus,stretched_exponential
from src.solvers import local_power_law
from src.visualization import metrics
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_siegert_and_derivative():
    m=np.array([1.,25.,1000.]);q=scattering_vector();b=.8
    np.testing.assert_allclose(msd_from_field(np.sqrt(correlation_forward(m,b)/b)),m,rtol=1e-12)
    t=np.array([2.,20.,100.]);h=1e-4;A,a,p=.8,.02,.7
    fd=(stretched_exponential(t+h,A,a,p)-stretched_exponential(t-h,A,a,p))/(2*h)
    np.testing.assert_allclose(fd,-a*p*t**(p-1)*stretched_exponential(t,A,a,p),rtol=1e-8)
def test_newtonian_and_elastic_limits():
    t=np.geomspace(1,1e5,70);eta=.003;a=250e-9;kT=1.38e-23*310.15
    m=kT*(t*1e-6)/(np.pi*a*eta)*1e18
    w,g1,g2=modulus(t,m,np.ones_like(t));np.testing.assert_allclose(g2,eta*w,rtol=1e-13);np.testing.assert_allclose(g1,0,atol=2e-12)
    w,g1,g2=modulus(t,np.full_like(t,100),np.zeros_like(t));np.testing.assert_allclose(g1,g1[0]);np.testing.assert_array_equal(g2,0)
def test_powerlaw_regression():
    t=np.geomspace(2,1e6,100);m=4*t**.62;s,a=local_power_law(t,m)
    np.testing.assert_allclose(s,m,rtol=1e-11);np.testing.assert_allclose(a,.62,atol=1e-11)
def test_known_metrics():
    a=np.array([0.,1.,2.]);b=np.array([0.,2.,2.]);ncc=6/(np.sqrt(5)*np.sqrt(8));nrmse=np.sqrt(1/3)/2
    np.testing.assert_allclose(metrics(a,b),[ncc,nrmse]);np.testing.assert_allclose(judge.agreement(a,b),[ncc,nrmse])
    with pytest.raises(ValueError):metrics(a,np.ones(3))
def command(path,report,cwd=None,script=None):
    return subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],capture_output=True,text=True,cwd=cwd)
@pytest.mark.parametrize('case',['missing','key','shape','dtype','complex','nan','inf','corrupt','zero'])
def test_judge_bad_inputs(tmp_path,case):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:d={k:z[k] for k in z.files}
    p=tmp_path/'prediction.npz';report=tmp_path/'nested/report.json'
    if case=='key':del d['msd']
    elif case=='shape':d['msd']=d['msd'].ravel()
    elif case=='dtype':d['msd']=d['msd'].astype(np.float32)
    elif case=='complex':d['msd']=d['msd'].astype(complex)
    elif case in ('nan','inf'):d['msd'][0,0]=float(case)
    elif case=='zero':d={k:np.zeros_like(v) for k,v in d.items()}
    if case=='corrupt':p.write_bytes(b'not npz')
    elif case!='missing':np.savez(p,**d)
    run=command(p,report,cwd=tmp_path);r=json.loads(report.read_text())
    assert run.returncode==(1 if case=='zero' else 2)
    assert r['status']==('fail' if case=='zero' else 'error') and not r['passed']
    if case!='zero':assert r['metrics']=={} and r['errors']
def test_judge_cwd_assets_write(tmp_path):
    p=ROOT/'output/reconstruction.npz';out=tmp_path/'ok.json';run=command(p,out,cwd=tmp_path);assert run.returncode==0,run.stderr
    fake=tmp_path/'fake';fake.mkdir();shutil.copy(ROOT/'evaluation/judge.py',fake/'judge.py');shutil.copy(ROOT/'evaluation/metrics.json',fake/'metrics.json')
    run=command(p,tmp_path/'err.json',script=fake/'judge.py');assert run.returncode==2
    blocker=tmp_path/'blocker';blocker.write_text('file');run=command(p,blocker/'report.json');assert run.returncode==2 and 'Cannot write report' in run.stderr
def test_public_metadata():
    from src import preprocessing as p
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['1.333','173','633','500','310.15','1.38e-23']:assert value in text
    assert (p.N,p.ANGLE_DEGREES,p.WAVELENGTH_NM,p.RADIUS_NM,p.TEMPERATURE_K,p.KB)==(1.333,173.,633.,250.,310.15,1.38e-23)
    assert 'github' not in text.lower()
