from pathlib import Path
import importlib.util,json,sys,subprocess,numpy as np,pytest
T=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('j',T/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','nan','inf'])
def test_invalid(tmp_path,kind):
    p=tmp_path/'bad.npz';a=np.ones((1,324))
    if kind=='shape':a=a[:,:2]
    if kind=='float32':a=a.astype('float32')
    if kind=='complex':a=a.astype(complex)
    if kind in ('nan','inf'):a.flat[0]=float(kind)
    if kind!='missing':np.savez_compressed(p,**{('wrong' if kind=='key' else 'reconstruction'):a})
    r=j.evaluate(p);assert r['status']=='error' and r['metrics']=={} and r['errors']
def test_poor_and_reference_priority(tmp_path):
    good=T/'evaluation/baseline_reference.npz';assert j.evaluate(good)['passed']
    bad=tmp_path/'bad.npz';np.savez_compressed(bad,reconstruction=np.ones((1,324)))
    assert j.evaluate(bad)['status']=='fail'
    (tmp_path/'metrics.json').write_bytes((T/'evaluation/metrics.json').read_bytes());(tmp_path/'baseline_reference.npz').write_bytes(good.read_bytes());(tmp_path/'ground_truth.npz').write_bytes(bad.read_bytes())
    assert j.evaluate(good,tmp_path)['status']=='fail'
    (tmp_path/'ground_truth.npz').write_bytes(b'bad');assert j.evaluate(good,tmp_path)['status']=='error'
def test_cli_cwd_report(tmp_path):
    r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'nested/report.json')],cwd=tmp_path,capture_output=True);assert r.returncode==0
    r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction','absent','--report',str(tmp_path)],capture_output=True);assert r.returncode==2
