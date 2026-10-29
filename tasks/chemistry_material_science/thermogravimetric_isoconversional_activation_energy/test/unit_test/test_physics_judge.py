from pathlib import Path
import importlib.util,numpy as np,pytest
from scipy.integrate import quad
from src.physics_model import temperature_integral,senum_yang,R
from src.preprocessing import polynomial_derivative
T=Path(__file__).resolve().parents[2];s=importlib.util.spec_from_file_location('judge',T/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
def test_integral():
    for e in [20.,45.,75.]:
        q=quad(lambda t:np.exp(-e/(R*t)),300,400,epsabs=1e-20)[0];assert np.isclose(temperature_integral(e,300,400),q,rtol=1e-10)
def test_derivative():
    t=np.linspace(0,10,80);np.testing.assert_allclose(polynomial_derivative(t,2+3*t+t*t),3+2*t,rtol=1e-9,atol=1e-9)
@pytest.mark.parametrize('kind',['missing','key','shape','integer','complex','float16','nan','inf','poor'])
def test_bad_inputs(tmp_path,kind):
    path=tmp_path/'x.npz';ref=np.load(T/'evaluation/baseline_reference.npz')['reconstruction'];x=ref.copy()
    if kind!='missing':
        if kind=='shape':x=x[:,:2]
        if kind in ['integer','complex','float16']:x=x.astype(dict(integer=int,complex=complex,float16=np.float16)[kind])
        if kind in ['nan','inf']:x.flat[0]=float(kind)
        if kind=='poor':x*=2
        np.savez_compressed(path,**{('wrong' if kind=='key' else 'reconstruction'):x})
    r=j.evaluate(path);assert not r['passed'];assert r['status']==('fail' if kind=='poor' else 'error')
def test_reference(tmp_path):
    assert j.evaluate(T/'evaluation/baseline_reference.npz')['passed'];assert j.evaluate(T/'evaluation/baseline_reference.npz',tmp_path)['status']=='error'
def test_truth_priority_and_cli(tmp_path):
    import shutil,subprocess,sys
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy2(T/'evaluation'/name,tmp_path/name)
    ref=np.load(tmp_path/'baseline_reference.npz')['reconstruction'];np.savez_compressed(tmp_path/'ground_truth.npz',reconstruction=ref*2)
    assert j.evaluate(tmp_path/'baseline_reference.npz',tmp_path)['status']=='fail'
    (tmp_path/'ground_truth.npz').write_bytes(b'broken')
    assert j.evaluate(tmp_path/'baseline_reference.npz',tmp_path)['status']=='error'
    r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction','missing.npz','--report',str(tmp_path/'nested/report.json')],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and (tmp_path/'nested/report.json').is_file()
    r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction','missing.npz','--report',str(tmp_path)],capture_output=True)
    assert r.returncode==2 and b'Report write failed' in r.stderr
