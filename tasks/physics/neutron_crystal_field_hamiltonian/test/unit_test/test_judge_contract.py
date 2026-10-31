"""Independent evaluator input, mathematical metric and CLI rejection cases."""
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('independent_judge', ROOT/'evaluation/judge.py')
judge = importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)

def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz', allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def test_reference_and_independent_forward():
    report, code = judge.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert code==0 and report['passed']
    assert report['metrics']['model0_forward_consistency']<1e-10
    assert report['metrics']['model1_forward_consistency']<1e-10
    assert report['metrics']['model0_spectrum_nrmse']==0

@pytest.mark.parametrize('case',['missing_key','shape','dtype','nan','inf','complex'])
def test_malformed(tmp_path,case):
    data=baseline()
    if case=='missing_key': del data['selected_model']
    elif case=='shape': data['reconstruction']=data['reconstruction'][0]
    elif case=='dtype': data['reconstruction']=data['reconstruction'].astype(np.float32)
    elif case=='complex': data['neutron_spectrum']=data['neutron_spectrum'].astype(np.complex128)
    else: data['reconstruction'][0,0,0]=np.nan if case=='nan' else np.inf
    path=tmp_path/'bad.npz'; np.savez(path,**data)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_missing_prediction_and_missing_assets(tmp_path):
    assert judge.evaluate(tmp_path/'absent.npz')[1]==2
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['errors'] and report['metrics']=={}

def test_invalid_threshold_configuration(tmp_path):
    config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf-8'))
    config['forward_consistency_boundary']=float('nan')
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf-8')
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path)
    assert code==2 and report['metrics']=={} and 'Invalid metric boundary' in report['errors'][0]

def test_truth_precedence_and_hidden_peak_contract(tmp_path):
    # A deliberately perturbed synthetic evaluator asset tests file precedence;
    # it is not an assertion of actual crystal-field truth for this real task.
    assets=tmp_path/'task/evaluation'; assets.mkdir(parents=True)
    (assets.parent/'data').mkdir()
    for name in ('metrics.json','baseline_reference.npz'):
        shutil.copyfile(ROOT/'evaluation'/name,assets/name)
    shutil.copyfile(ROOT/'data/raw_data.npz',assets.parent/'data/raw_data.npz')
    data=baseline(); data['reconstruction'][0,0,0]*=1.02
    np.savez(assets/'ground_truth.npz',**data)
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets)
    assert code==1
    assert report['metrics']['model0_parameter_relative_max']==pytest.approx(.02/1.02)
    data=baseline(); data['measured_inelastic_peaks_meV']=np.ones((1,1))
    np.savez(assets/'baseline_reference.npz',**data)
    report,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets)
    assert code==2 and report['metrics']=={} and 'hidden peak' in report['errors'][0]

def test_bad_spectra_rejected_and_nrmse_independent(tmp_path):
    data=baseline(); original=data['neutron_spectrum'].copy()
    data['neutron_spectrum']*=0
    path=tmp_path/'zero.npz'; np.savez(path,**data)
    report,code=judge.evaluate(path)
    assert code==1 and not report['passed']
    expected=np.sqrt(np.mean(original[0,0]**2))/np.ptp(original[0,0])
    assert report['metrics']['model0_spectrum_nrmse']==pytest.approx(expected)
    assert report['metrics']['model0_temperature0_observed_ncc']==0

def test_gauge_equivalence_and_row_order(tmp_path):
    data=baseline(); data['reconstruction'][0,:,[2,4]]*=-1
    path=tmp_path/'gauge.npz'; np.savez(path,**data)
    assert judge.evaluate(path)[1]==0
    data=baseline(); data['reconstruction']=data['reconstruction'][:,::-1].copy()
    np.savez(path,**data)
    assert judge.evaluate(path)[1]==1

def test_all_required_quantities_scored(tmp_path):
    for key in ('reconstruction','resolution_coefficients','selected_model'):
        data=baseline()
        if key=='reconstruction': data[key][0,0,11]*=2
        elif key=='resolution_coefficients': data[key][0,1]+=.01
        else: data[key][0]=1
        path=tmp_path/(key+'.npz'); np.savez(path,**data)
        assert judge.evaluate(path)[1]==1

def test_cli_cwd_and_report_failure(tmp_path):
    out=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(out)]
    completed=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert completed.returncode==0 and json.loads(out.read_text(encoding='utf-8'))['passed']
    command[-1]=str(tmp_path)
    completed=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert completed.returncode==2 and 'Cannot write report' in completed.stderr
