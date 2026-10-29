import importlib.util,tempfile,subprocess,sys
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def private_path():
    with tempfile.TemporaryDirectory(dir=ROOT/'output') as p:yield Path(p)
def valid():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return f['reconstruction'].copy()
def test_correct_reference_and_all_nine_individual_errors(private_path):
    path=private_path/'p.npz';np.savez(path,reconstruction=valid());assert judge.evaluate(path)['passed']
    for i in range(9):
        values=valid();values[0,i]*=1.2;np.savez(path,reconstruction=values)
        assert judge.evaluate(path)['status']=='fail'
@pytest.mark.parametrize('bad',['missing_key','shape','dtype','nan','inf','negative'])
def test_malformed(private_path,bad):
    values=valid()
    if bad=='shape':values=values[0]
    if bad=='dtype':values=values.astype(np.float32)
    if bad=='nan':values[0,0]=np.nan
    if bad=='inf':values[0,0]=np.inf
    if bad=='negative':values[0,0]=-1
    path=private_path/'p.npz';np.savez(path,**({'unknown':values} if bad=='missing_key' else {'reconstruction':values}))
    report=judge.evaluate(path);assert report['status']=='error' and report['metrics']=={} and report['errors']
def test_missing_assets_zero_reference_and_nonfinite_metric(private_path):
    path=private_path/'p.npz';np.savez(path,reconstruction=valid())
    assert judge.evaluate(path,private_path)['status']=='error'
    assert judge.evaluate(private_path/'missing.npz')['status']=='error'
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:ref={k:f[k] for k in f.files}
    ref['reconstruction'][0,0]=0
    with pytest.raises(ValueError,match='hidden'):judge.score({'reconstruction':valid()},ref)
def test_cli_cwd_and_reportwrite(private_path):
    path=private_path/'p.npz';np.savez(path,reconstruction=valid());report=private_path/'r.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)]
    assert subprocess.run(cmd,cwd=private_path,capture_output=True).returncode==0
    cmd[-1]=str(private_path);assert subprocess.run(cmd,cwd=private_path,capture_output=True).returncode==2

def test_actual_truth_precedes_method_parameters_and_reference_dtype(private_path):
    import shutil
    for name in ['baseline_reference.npz','metrics.json']:shutil.copyfile(ROOT/'evaluation'/name,private_path/name)
    path=private_path/'p.npz';np.savez(path,reconstruction=valid())
    np.savez(private_path/'ground_truth.npz',reconstruction=valid()*1.2)
    report=judge.evaluate(path,private_path)
    assert report['status']=='fail'
    assert report['metrics']['maximum_kappa_relative_error']==pytest.approx(1/6)
    np.savez(private_path/'ground_truth.npz',reconstruction=valid().astype('float32'))
    report=judge.evaluate(path,private_path);assert report['status']=='error' and report['metrics']=={}
