from pathlib import Path
import copy,json,subprocess,sys,tempfile,shutil
import numpy as np
import pytest
from evaluation.judge import HERE,evaluate
import evaluation.judge as judge
@pytest.fixture
def good():
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:
        parameters=np.array([z[f'parameters_{a}_20261029'][0] for a in [0,45,90]])
        result={f'reconstruction_{a}':z[f'response_{a}_20261029'].copy() for a in [0,45,90]}
        result.update(magnetic_field=parameters[None,:,0]*1e-4,temperature=parameters[None,:,1]+273.15,polarization=parameters[None,:,2])
    return result
@pytest.mark.parametrize('key',['reconstruction_0','reconstruction_45','reconstruction_90','magnetic_field','temperature','polarization'])
def test_each_output_scores(good,key):
    assert evaluate(good)['passed']
    good[key].flat[0]+=.1 if key!='temperature' else 1.
    assert evaluate(good)['status']=='fail'
@pytest.mark.parametrize('fault',['missing','shape','dtype','nan','inf','zero_reference','bad_boundary'])
def test_invalid_input_evaluator(good,fault):
    kwargs={}
    if fault=='missing':good.pop('magnetic_field')
    if fault=='shape':good['magnetic_field']=good['magnetic_field'][0]
    if fault=='dtype':good['temperature']=good['temperature'].astype('float32')
    if fault in ['nan','inf']:good['reconstruction_0'][0,0]=float(fault)
    if fault=='zero_reference':
        with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        reference['parameters_0_20261029'][0,0]=0;kwargs['reference']=reference
    if fault=='bad_boundary':
        config=json.loads((HERE/'metrics.json').read_text());config['spectral_rmse_boundary']=float('inf');kwargs['configuration']=config
    r=evaluate(good,**kwargs);assert r['status']=='error' and r['metrics']=={} and r['errors'];json.dumps(r,allow_nan=False)
def test_native_cwd_missing_assets_and_reportwrite():
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as directory:
        work=Path(directory);report=work/'nested/report.json';cmd=[sys.executable,str(HERE/'judge.py'),'--prediction',str(HERE.parent/'output/reconstruction.npz'),'--report',str(report)]
        r=subprocess.run(cmd,cwd=work,capture_output=True);assert r.returncode==0 and json.loads(report.read_text())['passed']
        cmd[3]=str(work/'absent.npz');r=subprocess.run(cmd,cwd=work,capture_output=True);assert r.returncode==2 and json.loads(report.read_text())['metrics']=={}
        cmd[3]=str(HERE.parent/'output/reconstruction.npz');shutil.copyfile(HERE/'judge.py',work/'judge.py');cmd[1]=str(work/'judge.py');r=subprocess.run(cmd,cwd=work,capture_output=True);assert r.returncode==2 and json.loads(report.read_text())['status']=='error'
        cmd[-1]=str(work);r=subprocess.run(cmd,cwd=work,capture_output=True);assert r.returncode==2 and b'report write failed' in r.stderr
def test_actual_truth_precedes_method_reference(good,monkeypatch):
    config=json.loads((HERE/'metrics.json').read_text())
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as directory:
        work=Path(directory);np.savez(work/'ground_truth.npz',**good);np.savez(work/'baseline_reference.npz',wrong=np.array([0.]))
        monkeypatch.setattr(judge,'HERE',work)
        assert judge.evaluate(good,configuration=config,observations=observations)['passed']
def test_finite_prediction_overflow_returns_error(good):
    good['reconstruction_0'].fill(1e308)
    with np.errstate(over='ignore',invalid='ignore'):report=evaluate(good)
    assert report['status']=='error' and report['metrics']=={}
    json.dumps(report,allow_nan=False)
