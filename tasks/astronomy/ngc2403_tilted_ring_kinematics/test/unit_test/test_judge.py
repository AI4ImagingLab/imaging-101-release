from pathlib import Path
import importlib.util,json
import subprocess,sys
import numpy as np
import pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_kinematics_judge',T/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def known(offset=0):
    return dict(rotation=np.linspace(1,3,41)[None]+offset,dispersion=np.linspace(2,6,41)[None]+offset,position_angle=np.full((1,41),12.+offset))


def configuration():
    return json.loads((T/'evaluation/metrics.json').read_text())


def test_known_range_error_and_wrapped_degrees():
    ref=dict(rotation=np.array([1.,3.]),dispersion=np.array([2.,6.]),position_angle=np.array([179.,-179.]))
    est=dict(rotation=np.array([1.,1.]),dispersion=np.array([2.,2.]),position_angle=np.array([-179.,179.]))
    values=judge.profile_metrics(est,ref)
    assert values['rotation_range_nrmse']==pytest.approx(1/np.sqrt(2))
    assert values['dispersion_range_nrmse']==pytest.approx(1/np.sqrt(2))
    assert values['position_angle_wrapped_rms_deg']==2


def test_actual_truth_has_priority_even_with_invalid_baseline(tmp_path):
    truth=known(4)
    np.savez(tmp_path/'ground_truth.npz',**truth)
    (tmp_path/'baseline_reference.npz').write_bytes(b'invalid baseline must not be opened')
    loaded=judge.load_reference(tmp_path)
    for key in judge.KEYS:np.testing.assert_array_equal(loaded[key],truth[key])


def test_existing_invalid_truth_never_falls_back(tmp_path):
    np.savez(tmp_path/'baseline_reference.npz',**known())
    np.savez(tmp_path/'ground_truth.npz',rotation=np.ones((1,41)))
    with pytest.raises(ValueError,match='Missing required key'):judge.load_reference(tmp_path)


@pytest.mark.parametrize('value',[True,'0.1',-.1,float('nan'),float('inf')])
def test_bad_configuration_rejected(value,tmp_path):
    config=configuration();config['rotation_range_nrmse_boundary']=value
    (tmp_path/'metrics.json').write_text(json.dumps(config))
    report,code=judge.evaluate(tmp_path/'missing.npz',reference_dir=tmp_path)
    assert code==2 and report['status']=='error' and report['metrics']=={}


@pytest.mark.parametrize('case',['missing_key','wrong_shape','integer','complex','object','nan','inf','negative'])
def test_malformed_prediction(case,tmp_path):
    arrays=known()
    if case=='missing_key':arrays.pop('dispersion')
    elif case=='wrong_shape':arrays['dispersion']=np.ones(41)
    elif case=='integer':arrays['dispersion']=np.ones((1,41),dtype=int)
    elif case=='complex':arrays['dispersion']=np.ones((1,41),dtype=complex)
    elif case=='object':arrays['dispersion']=np.array([['x']*41],dtype=object)
    elif case=='nan':arrays['dispersion'][0,0]=np.nan
    elif case=='inf':arrays['dispersion'][0,0]=np.inf
    else:arrays['dispersion'][0,0]=-1
    path=tmp_path/'submission.npz';np.savez(path,**arrays)
    report,code=judge.evaluate(path)
    assert code==2 and not report['passed'] and report['metrics']=={} and report['errors']


def test_finite_extreme_values_cannot_emit_nonfinite_metrics(tmp_path):
    arrays=known();arrays['rotation'].fill(1e308)
    path=tmp_path/'extreme.npz';np.savez(path,**arrays)
    report,code=judge.evaluate(path)
    assert code==2 and report['metrics']=={}
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize('kind',['zero','uniform','shuffle'])
def test_poor_complete_outputs_fail_all_field_contract(kind,tmp_path):
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in judge.KEYS}
    if kind=='zero':
        for value in arrays.values():value.fill(0)
    elif kind=='uniform':
        for value in arrays.values():value.fill(value.mean())
    else:
        for key in ['rotation','dispersion']:arrays[key]=arrays[key][:,np.random.default_rng(17).permutation(41)]
    path=tmp_path/'poor.npz';np.savez(path,**arrays)
    report,code=judge.evaluate(path)
    assert code==1 and report['status']=='fail' and not report['passed']
    assert set(report['metrics'])=={'rotation_range_nrmse','dispersion_range_nrmse','position_angle_wrapped_rms_deg'}


def test_constant_reference_is_an_evaluator_error():
    ref=known();ref['rotation'].fill(1)
    with pytest.raises(ValueError,match='Degenerate'):judge.profile_metrics(known(),ref)


def test_foreign_directory_cli_fail_and_error_and_report_failure(tmp_path):
    poor=known();prediction=tmp_path/'poor.npz';np.savez(prediction,**poor)
    report=tmp_path/'nested/report.json'
    command=[sys.executable,'-X','utf8',str(T/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    process=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',shell=False)
    assert process.returncode==1
    content=json.loads(report.read_text(encoding='utf-8'))
    assert content['status']=='fail' and not content['passed'] and not content['errors']
    missing=tmp_path/'missing.npz'
    command[command.index('--prediction')+1]=str(missing)
    process=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',shell=False)
    assert process.returncode==2 and json.loads(report.read_text(encoding='utf-8'))['metrics']=={}
    parent=tmp_path/'not_a_directory';parent.write_text('file',encoding='utf-8')
    command[command.index('--report')+1]=str(parent/'report.json')
    process=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',shell=False)
    assert process.returncode==2 and 'Cannot write Judge report' in process.stderr
