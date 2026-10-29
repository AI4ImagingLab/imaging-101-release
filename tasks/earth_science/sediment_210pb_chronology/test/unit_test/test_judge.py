import importlib.util,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def private_path():
    with tempfile.TemporaryDirectory(dir=ROOT/'output') as name:yield Path(name)

def prediction():
    with np.load(ROOT/'notebooks/reference_outputs/upstream_profiles.npz') as f:p=f['profile']
    return {'reconstruction':p[None,2],'sedimentation_rate':p[None,1],'initial_activity':p[None,0],'modeled_activity':p[None,3]}

def test_known_correct_original_and_bad_chronology(private_path):
    path=private_path/'p.npz';p=prediction();np.savez(path,**p)
    assert judge.evaluate(path)['passed']
    p['reconstruction']=p['reconstruction']*2
    p['sedimentation_rate']=p['sedimentation_rate']/2
    np.savez(path,**p);result=judge.evaluate(path)
    assert result['status']=='fail' and not result['passed']

@pytest.mark.parametrize('key',['sedimentation_rate','initial_activity','modeled_activity'])
def test_each_additional_physical_quantity_is_scored(private_path,key):
    p=prediction();p[key]=p[key]*1.01
    path=private_path/'p.npz';np.savez(path,**p)
    result=judge.evaluate(path)
    assert result['status']=='fail' and not result['passed']

@pytest.mark.parametrize('bad',['key','shape','dtype','nan','inf'])
def test_malformed(private_path,bad):
    p=prediction()
    if bad=='key':del p['initial_activity']
    if bad=='shape':p['reconstruction']=p['reconstruction'][0]
    if bad=='dtype':p['reconstruction']=p['reconstruction'].astype(np.float32)
    if bad=='nan':p['reconstruction'][0,0]=np.nan
    if bad=='inf':p['initial_activity'][0,0]=np.inf
    path=private_path/'bad.npz';np.savez(path,**p);r=judge.evaluate(path)
    assert r['status']=='error' and r['metrics']=={} and r['errors']

def test_missing_evaluator_assets_and_zero_truth(private_path):
    path=private_path/'p.npz';np.savez(path,**prediction())
    assert judge.evaluate(private_path/'missing.npz')['status']=='error'
    assert judge.evaluate(path,private_path)['status']=='error'
    truth={'reconstruction':np.zeros((1,14))}
    with np.load(ROOT/'data/raw_data.npz') as f:obs=[f[k][0] for k in ['mass_depth','activity','activity_error']]
    with pytest.raises(ValueError,match='hidden'):judge.score(prediction(),truth,*obs)

def test_cli_cwd_independence_exit_and_report_write(private_path):
    path=private_path/'p.npz';np.savez(path,**prediction());report=private_path/'r.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)]
    assert subprocess.run(cmd,cwd=private_path,capture_output=True).returncode==0
    cmd[-1]=str(private_path)
    assert subprocess.run(cmd,cwd=private_path,capture_output=True).returncode==2
