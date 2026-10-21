from pathlib import Path
import importlib.util,json,subprocess,sys,shutil,os
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_metric_known_cases():
    assert judge.relative_error(np.array([2.,4.]),np.array([1.,2.]))==1
    assert judge.relative_error(np.array([1j]),np.array([1j]))==0
    with pytest.raises(ValueError):judge.relative_error(np.zeros(2),np.zeros(2))
def invoke(prediction,report,cwd=None,script=None):
    return subprocess.run([sys.executable,str(script or ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=cwd,capture_output=True,text=True,encoding='utf8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
def sample():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def test_pipeline_and_foreign_cwd(tmp_path):
    from src.main import run
    output=tmp_path/'fresh.npz';run(ROOT/'data/raw_data.npz',output)
    report=tmp_path/'nested/report.json';p=invoke(output,report,cwd=tmp_path);assert p.returncode==0,p.stderr
    data=json.loads(report.read_text());assert data['status']=='pass' and data['passed'];assert data['metrics']
@pytest.mark.parametrize('defect',['missing_file','missing_key','shape','dtype','nan','infinity','complex_dtype'])
def test_invalid_prediction(defect,tmp_path):
    a=sample();key=next(iter(a));path=tmp_path/'input.npz'
    if defect=='missing_key':a.pop(key)
    elif defect=='shape':a[key]=a[key].reshape(-1)
    elif defect=='dtype':a[key]=a[key].astype('complex64' if np.iscomplexobj(a[key]) else 'float32')
    elif defect=='complex_dtype':a[key]=a[key].astype('float64' if np.iscomplexobj(a[key]) else 'complex128')
    elif defect in ['nan','infinity']:a[key].flat[0]=np.nan if defect=='nan' else np.inf
    if defect!='missing_file':np.savez(path,**a)
    report=tmp_path/'report.json';p=invoke(path,report);assert p.returncode==2
    r=json.loads(report.read_text());assert r['status']=='error' and not r['passed'] and r['metrics']=={} and r['errors']
def test_every_required_quantity_matters(tmp_path):
    for key in sample():
        a=sample();a[key]=np.zeros_like(a[key]);path=tmp_path/'poor.npz';np.savez(path,**a);report=tmp_path/'poor.json';p=invoke(path,report)
        assert p.returncode==1,(key,p.stdout,p.stderr);assert json.loads(report.read_text())['status']=='fail'
def test_missing_evaluator_assets_and_truth_priority(tmp_path):
    script=tmp_path/'judge.py';shutil.copy2(ROOT/'evaluation/judge.py',script);shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    pred=tmp_path/'valid.npz';np.savez(pred,**sample());report=tmp_path/'report.json'
    assert invoke(pred,report,script=script).returncode==2
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    np.savez(tmp_path/'ground_truth.npz',**{k:2*v for k,v in sample().items()})
    assert invoke(pred,report,script=script).returncode==1
def test_report_write_failure(tmp_path):
    occupied=tmp_path/'occupied';occupied.write_text('file')
    p=invoke(ROOT/'evaluation/baseline_reference.npz',occupied/'report.json');assert p.returncode==2 and 'Cannot write report' in p.stderr
def test_main_import_preserves_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert matplotlib.get_backend()==before

@pytest.mark.parametrize('defect',['no_metrics','nonfinite_boundary','negative_boundary','missing_output_coverage'])
def test_undefined_metric_configuration_rejected(defect,tmp_path):
    script=tmp_path/'judge.py';shutil.copy2(ROOT/'evaluation/judge.py',script);config=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if defect=='no_metrics':config['metric_groups']=[]
    elif defect=='nonfinite_boundary':config['metric_groups'][0]['boundary']=float('inf')
    elif defect=='negative_boundary':config['metric_groups'][0]['boundary']=-1
    else:config['metric_groups'][0]['key']='unknown_output'
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8');shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz');pred=tmp_path/'prediction.npz';np.savez(pred,**sample());report=tmp_path/'report.json';p=invoke(pred,report,script=script);assert p.returncode==2;data=json.loads(report.read_text());assert data['metrics']=={} and data['errors']
