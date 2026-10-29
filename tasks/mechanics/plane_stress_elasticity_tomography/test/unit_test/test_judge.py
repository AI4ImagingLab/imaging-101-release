import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('qsei_independent_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def assets(tmp_path):
    task=tmp_path/'task';evaluation=task/'evaluation';evaluation.mkdir(parents=True)
    (task/'data').mkdir()
    for name in ('ground_truth.npz','metrics.json'):shutil.copyfile(ROOT/'evaluation'/name,evaluation/name)
    shutil.copyfile(ROOT/'data/raw_data.npz',task/'data/raw_data.npz')
    return evaluation

@pytest.fixture
def prediction(tmp_path):
    with np.load(ROOT/'notebooks/reference_outputs/qualified_tv.npz',allow_pickle=False) as data:field=data['reconstruction']
    path=tmp_path/'prediction.npz';np.savez_compressed(path,reconstruction=field)
    return path

def test_independent_metric_conventions_known_vectors():
    reference=np.arange(1,801,dtype=np.float64);estimate=reference+1
    metrics=judge.array_metrics(estimate,reference)
    assert metrics['nrmse_vs_ref']==1/799
    assert metrics['ncc_vs_ref']==estimate@reference/np.sqrt((estimate@estimate)*(reference@reference))

def test_source_result_pass_and_full_report(assets,prediction):
    report,code=judge.evaluate(prediction,assets)
    assert code==0 and report['status']=='pass' and report['passed']
    assert len(report['metrics'])==4 and not report['errors']
    assert set(report)=={'status','passed','metrics','thresholds','errors'}

@pytest.mark.parametrize('kind',['missing_key','extra_key','shape','float32','complex','nan','inf','zero','negative','object','corrupt','missing'])
def test_malformed_submissions_have_empty_metrics(assets,tmp_path,kind):
    path=tmp_path/'bad.npz';array=np.ones((1,800),np.float64)
    if kind=='missing_key':np.savez_compressed(path,other=array)
    elif kind=='extra_key':np.savez_compressed(path,reconstruction=array,other=array)
    elif kind=='shape':np.savez_compressed(path,reconstruction=array.T)
    elif kind=='float32':np.savez_compressed(path,reconstruction=array.astype(np.float32))
    elif kind=='complex':np.savez_compressed(path,reconstruction=array.astype(np.complex128))
    elif kind in ('nan','inf','zero','negative'):
        array[0,0]={'nan':np.nan,'inf':np.inf,'zero':0.,'negative':-1.}[kind]
        np.savez_compressed(path,reconstruction=array)
    elif kind=='object':np.savez_compressed(path,reconstruction=array.astype(object))
    elif kind=='corrupt':path.write_bytes(b'not an npz file')
    report,code=judge.evaluate(path,assets)
    assert code==2 and report['status']=='error' and not report['passed']
    assert report['metrics']=={} and report['errors']

@pytest.mark.parametrize('kind',['homogeneous','shuffled','inverted'])
def test_scientifically_poor_positive_fields_fail(assets,tmp_path,kind):
    truth=np.load(assets/'ground_truth.npz',allow_pickle=False)['reconstruction']
    if kind=='homogeneous':field=np.full((1,800),236.25)
    elif kind=='shuffled':field=truth[:,np.random.default_rng(17).permutation(800)]
    else:field=301-truth
    path=tmp_path/f'{kind}.npz';np.savez_compressed(path,reconstruction=field)
    report,code=judge.evaluate(path,assets)
    assert code==1 and report['status']=='fail' and report['metrics']

def test_actual_truth_priority_and_legitimate_baseline_fallback(assets,prediction):
    truth=np.load(assets/'ground_truth.npz',allow_pickle=False)['reconstruction']
    np.savez_compressed(assets/'baseline_reference.npz',reconstruction=301-truth)
    assert judge.evaluate(prediction,assets)[1]==0
    np.savez_compressed(assets/'ground_truth.npz',reconstruction=np.ones((800,1)))
    assert judge.evaluate(prediction,assets)[1]==2
    np.savez_compressed(assets/'baseline_reference.npz',reconstruction=truth)
    (assets/'ground_truth.npz').unlink()
    assert judge.evaluate(prediction,assets)[1]==0

@pytest.mark.parametrize('kind',['references','metrics','calibration','constant_reference','bad_json','list_json','nan_json'])
def test_unavailable_or_undefined_evaluator_assets(assets,prediction,kind):
    if kind=='references':(assets/'ground_truth.npz').unlink()
    elif kind=='metrics':(assets/'metrics.json').unlink()
    elif kind=='calibration':(assets.parent/'data/raw_data.npz').unlink()
    elif kind=='constant_reference':np.savez_compressed(assets/'ground_truth.npz',reconstruction=np.ones((1,800)))
    elif kind=='bad_json':(assets/'metrics.json').write_text('{',encoding='utf8')
    elif kind=='list_json':(assets/'metrics.json').write_text('[]',encoding='utf8')
    else:(assets/'metrics.json').write_text('{"ncc_boundary":NaN}',encoding='utf8')
    report,code=judge.evaluate(prediction,assets)
    assert code==2 and report['metrics']=={} and report['errors']

@pytest.mark.parametrize('metric,key,comparison',[
 ('ncc_vs_ref','ncc_boundary','>='),('nrmse_vs_ref','nrmse_boundary','<='),
 ('relative_displacement_error','relative_displacement_error_boundary','<='),
 ('maximum_abs_principal_strain','maximum_abs_principal_strain_boundary','<=')])
def test_each_boundary_is_actually_consumed_near_measured_value(assets,prediction,metric,key,comparison):
    measured=judge.evaluate(prediction,assets)[0]['metrics'][metric]
    config=json.loads((assets/'metrics.json').read_text())
    config.update(ncc_boundary=-1.,nrmse_boundary=10.,relative_displacement_error_boundary=10.,maximum_abs_principal_strain_boundary=1.)
    for should_pass in (True,False):
        sign=-1 if (should_pass==(comparison=='>=')) else 1
        config[key]=measured+sign*max(abs(measured)*1e-6,1e-9)
        (assets/'metrics.json').write_text(json.dumps(config),encoding='utf8')
        report,code=judge.evaluate(prediction,assets)
        assert code==(0 if should_pass else 1)
        assert report['thresholds'][metric]['boundary']==config[key]

def test_standalone_other_working_directory_and_report_failure(tmp_path,prediction):
    report=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0 and json.loads(report.read_text())['passed']
    blocked=tmp_path/'blocked';blocked.write_text('file',encoding='utf8')
    command[-1]=str(blocked/'report.json')
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==2 and 'Unable to write Judge report' in result.stderr
