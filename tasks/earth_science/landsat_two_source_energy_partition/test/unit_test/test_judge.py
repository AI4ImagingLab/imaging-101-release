import copy,json,subprocess,sys,tempfile,shutil
from pathlib import Path
import numpy as np
import pytest
from evaluation.judge import HERE,evaluate
import evaluation.judge as judge
@pytest.fixture
def good():
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}
@pytest.mark.parametrize('key',['reconstruction','soil_temperature','canopy_temperature','soil_heat_flux','radiation'])
def test_every_quantity_is_scored(good,key):
    assert evaluate(good)['passed'];good[key].flat[0]+=1
    assert evaluate(good)['status']=='fail'
@pytest.mark.parametrize('fault',['missing','shape','dtype','nan','inf','zero_kelvin','zero_reference','boundary'])
def test_invalid_contract_and_evaluator(good,fault):
    kwargs={}
    if fault=='missing':good.pop('radiation')
    if fault=='shape':good['reconstruction']=good['reconstruction'][0]
    if fault=='dtype':good['radiation']=good['radiation'].astype('float32')
    if fault in ['nan','inf']:good['soil_temperature'][0,0]=float(fault)
    if fault=='zero_kelvin':good['soil_temperature'][0,0]=0
    if fault=='zero_reference':
        reference=copy.deepcopy(good);reference['canopy_temperature']*=0;kwargs['reference']=reference
    if fault=='boundary':
        config=json.loads((HERE/'metrics.json').read_text());config['maximum_flux_absolute_error_boundary']=float('inf');kwargs['configuration']=config
    report=evaluate(good,**kwargs);assert report['status']=='error' and report['metrics']=={} and report['errors'];json.dumps(report,allow_nan=False)
def test_native_cwd_missing_files_assets_and_report_write():
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as name:
        work=Path(name);report=work/'nested/report.json';command=[sys.executable,str(HERE/'judge.py'),'--prediction',str(HERE.parent/'output/reconstruction.npz'),'--report',str(report)]
        result=subprocess.run(command,cwd=work,capture_output=True);assert result.returncode==0 and json.loads(report.read_text())['passed']
        command[3]=str(work/'absent.npz');result=subprocess.run(command,cwd=work,capture_output=True);assert result.returncode==2 and json.loads(report.read_text())['metrics']=={}
        command[3]=str(HERE.parent/'output/reconstruction.npz');shutil.copyfile(HERE/'judge.py',work/'judge.py');command[1]=str(work/'judge.py');result=subprocess.run(command,cwd=work,capture_output=True);assert result.returncode==2 and json.loads(report.read_text())['status']=='error'
        command[-1]=str(work);result=subprocess.run(command,cwd=work,capture_output=True);assert result.returncode==2 and b'report write failed' in result.stderr
def test_actual_truth_precedes_method_reference(good,monkeypatch):
    config=json.loads((HERE/'metrics.json').read_text())
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
    with tempfile.TemporaryDirectory(dir=HERE.parent/'output') as name:
        work=Path(name);np.savez(work/'ground_truth.npz',**good);np.savez(work/'baseline_reference.npz',wrong=np.array([0.]))
        monkeypatch.setattr(judge,'HERE',work)
        assert judge.evaluate(good,configuration=config,observations=observations)['passed']
def test_finite_prediction_overflow_is_error_json(good):
    good['soil_temperature'].fill(1e308)
    with np.errstate(over='ignore',invalid='ignore'):report=evaluate(good)
    assert report['status']=='error' and report['metrics']=={}
    json.dumps(report,allow_nan=False)
