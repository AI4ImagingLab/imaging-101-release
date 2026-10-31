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
        a=sample();a[key]=np.zeros_like(a[key]);
        if key=='lower_residual':a[key].fill(1e6)
        if key=='upper_residual':a[key].fill(1.)
        path=tmp_path/'poor.npz';np.savez(path,**a);report=tmp_path/'poor.json';p=invoke(path,report)
        assert p.returncode==(2 if key in ('natural_frequency','damping_ratio') else 1),(key,p.stdout,p.stderr);assert not json.loads(report.read_text())['passed']
def test_missing_evaluator_assets_and_truth_priority(tmp_path):
    script=tmp_path/'judge.py';shutil.copy2(ROOT/'evaluation/judge.py',script);shutil.copy2(ROOT/'evaluation/metrics.json',tmp_path/'metrics.json')
    pred=tmp_path/'valid.npz';np.savez(pred,**sample());report=tmp_path/'report.json'
    assert invoke(pred,report,script=script).returncode==2
    shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz')
    np.savez(tmp_path/'ground_truth.npz',**{k:2*v for k,v in sample().items()})
    assert invoke(pred,report,script=script).returncode==2
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
    if defect=='no_metrics':config.pop('maximum_response_mse_excess_boundary')
    elif defect=='nonfinite_boundary':config['maximum_response_mse_excess_boundary']=float('inf')
    elif defect=='negative_boundary':config['maximum_response_mse_excess_boundary']=-1
    else:config['output_schema']['unknown_output']=config['output_schema'].pop('reconstruction')
    (tmp_path/'metrics.json').write_text(json.dumps(config),encoding='utf8');shutil.copy2(ROOT/'evaluation/baseline_reference.npz',tmp_path/'baseline_reference.npz');pred=tmp_path/'prediction.npz';np.savez(pred,**sample());report=tmp_path/'report.json';p=invoke(pred,report,script=script);assert p.returncode==2;data=json.loads(report.read_text());assert data['metrics']=={} and data['errors']


def test_known_accelerance_residual_units_and_signs():
    a=sample();a['reconstruction']*=0;a['lower_residual'][:]=2+3j;a['upper_residual'][:]=4+5j
    f=np.array([0.,1.,2.]);expected=2+3j-(2*np.pi*f)**2*(4+5j)
    np.testing.assert_allclose(judge.predicted_accelerance(a,f),np.broadcast_to(expected,(6,3)),rtol=1e-14)


def test_independent_residue_variable_projection_improvement(tmp_path):
    a=sample()
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as raw:f=raw['frequency'][0];y=raw['accelerance'][0]
    start=np.argmin(abs(f-10));end=np.argmin(abs(f-5000));w=2*np.pi*f[start:end]
    pole=2*np.pi*a['natural_frequency'][0]*(-a['damping_ratio'][0]+1j*np.sqrt(1-a['damping_ratio'][0]**2))
    positive=-w[:,None]**2/(1j*w[:,None]-pole);negative=-w[:,None]**2/(1j*w[:,None]-pole.conjugate())
    columns=np.column_stack([positive+negative,1j*(positive-negative),np.ones(len(w)),1j*np.ones(len(w)),-w**2,-1j*w**2])
    matrix=np.vstack([columns.real,columns.imag]);scales=np.linalg.norm(matrix,axis=0)
    target=np.vstack([y[:,start:end].T.real,y[:,start:end].T.imag])
    coefficients=(np.linalg.lstsq(matrix/scales,target,rcond=None)[0]/scales[:,None]).T
    before=np.mean(abs(judge.predicted_accelerance(a,f[start:end])-y[:,start:end])**2,axis=1)
    a['reconstruction']=(coefficients[:,:7]+1j*coefficients[:,7:14])[None]
    a['lower_residual']=(coefficients[:,14]+1j*coefficients[:,15])[None]
    a['upper_residual']=(coefficients[:,16]+1j*coefficients[:,17])[None]
    after=np.mean(abs(judge.predicted_accelerance(a,f[start:end])-y[:,start:end])**2,axis=1)
    assert np.sum(after)<=np.sum(before)*(1+1e-12)
    outgrid=np.r_[np.arange(0,f[0],f[1]-f[0]),f[:end]];a['fitted_accelerance']=judge.predicted_accelerance(a,outgrid)[None]
    path=tmp_path/'variable_projection.npz';np.savez(path,**a);report=judge.evaluate(path)
    assert report['passed'],report
