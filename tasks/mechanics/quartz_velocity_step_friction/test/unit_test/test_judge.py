from pathlib import Path
import numpy as np,json,subprocess,sys,shutil,importlib.util,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_independent_known_metrics():
    ref={'reconstruction':np.array([[.01,.011,3e-6]]),'state':np.ones((1,914)),'slider_velocity':np.ones((1,914))*1e-6,'friction':np.ones((1,914))*.7}
    pred={k:v.copy() for k,v in ref.items()};pred['reconstruction']+=np.array([[.001,.002,1e-7]]);pred['state']+=.01;pred['slider_velocity']*=1.1;pred['friction']+=.0001
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());m,_=judge.score(pred,ref,ref['friction'][0],cfg)
    for key,value in [('a_absolute_error',.001),('b_absolute_error',.002),('Dc_absolute_error_m',1e-7),('state_maximum_absolute_error_s',.01),('velocity_relative_l2_error',.1),('friction_maximum_absolute_error',.0001),('observed_friction_RMS',.0001)]:assert m[key]==pytest.approx(value)

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','poor','overflow','nonpositive'])
def test_strict_contract(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:values={k:f[k] for k in f.files}
    if kind=='key':del values['state']
    if kind=='shape':values['state']=values['state'][0]
    if kind=='dtype':values['state']=values['state'].astype('float32')
    if kind=='complex':values['state']=values['state'].astype('complex128')
    if kind in ['nan','inf']:values['state'].flat[0]=float(kind)
    if kind=='poor':values['reconstruction']*=2
    if kind=='overflow':values['friction'].fill(1e308)
    if kind=='nonpositive':values['slider_velocity'].flat[0]=0
    p=tmp_path/'p.npz';r=tmp_path/'sub/r.json'
    if kind!='missing':np.savez(p,**values)
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True)
    report=json.loads(r.read_text());assert result.returncode==(1 if kind=='poor' else 2)
    if kind!='poor':assert report['metrics']=={} and report['errors']

def test_assets_precedence_and_write_failure(tmp_path):
    shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');shutil.copytree(ROOT/'data',tmp_path/'data')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:values={k:f[k] for k in f.files}
    np.savez(tmp_path/'p.npz',**values);values['reconstruction']*=2;np.savez(tmp_path/'evaluation/ground_truth.npz',**values)
    command=[sys.executable,str(tmp_path/'evaluation/judge.py'),'--prediction',str(tmp_path/'p.npz'),'--report',str(tmp_path/'r.json')]
    assert subprocess.run(command,capture_output=True).returncode==1
    (tmp_path/'evaluation/ground_truth.npz').unlink();(tmp_path/'evaluation/baseline_reference.npz').unlink()
    assert subprocess.run(command,capture_output=True).returncode==2
    result=subprocess.run(command[:-1]+[str(tmp_path)],capture_output=True)
    assert result.returncode==2 and b'Report-write failure' in result.stderr
