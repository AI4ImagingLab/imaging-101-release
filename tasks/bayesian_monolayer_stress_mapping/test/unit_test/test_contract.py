from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_stress_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture(scope='module')
def truth():
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as f:return f['reconstruction'].copy()
def save(tmp_path,value=None):
    p=tmp_path/'candidate.npz'
    np.savez_compressed(p,**({'reconstruction':value} if value is not None else {'other':np.zeros(1)}))
    return p
def test_metric_known_cases(truth):
    value=judge.metrics(truth,truth)
    assert all(v==pytest.approx(1) if k.endswith('ncc') else v==0 for k,v in value.items())
    small=np.tile(np.array([1.,2.,3.]),(1,3,1,1));reference=np.tile(np.array([1.,2.,4.]),(1,3,1,1))
    result=judge.metrics(small,reference)
    assert result['xy_ncc']==pytest.approx(17/np.sqrt(294))
    assert result['xx_nrmse']==pytest.approx(np.sqrt(1/3)/3)
@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','nan','inf','object','overflow'])
def test_invalid_prediction(tmp_path,truth,kind):
    x=truth.copy()
    if kind=='missing':path=tmp_path/'missing.npz'
    elif kind=='key':path=save(tmp_path)
    else:
        if kind=='shape':x=x[:,0]
        if kind=='float32':x=x.astype(np.float32)
        if kind=='complex':x=x.astype(complex)
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        if kind=='object':x=np.array([object()],dtype=object)
        if kind=='overflow':x[:]=1e308
        path=save(tmp_path,x)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
@pytest.mark.parametrize('component',[0,1,2])
def test_each_component_is_mandatory(tmp_path,truth,component):
    x=truth.copy();x[:,component]*=-1
    report,code=judge.evaluate(save(tmp_path,x))
    assert code==1 and not report['passed']
    assert any(judge.COMPONENTS[component] in error for error in report['errors'])
def test_zero_and_shifted_outputs_fail(tmp_path,truth):
    for x in [np.zeros_like(truth),truth[...,::-1],truth*3,truth+3]:
        report,code=judge.evaluate(save(tmp_path,x));assert code==1
def test_foreign_directory_and_report_parents(tmp_path,truth):
    prediction=save(tmp_path,truth);report=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(report.read_text())['passed']
def test_missing_hidden_assets(tmp_path,truth):
    previous=judge.ASSETS;judge.ASSETS=tmp_path
    try:
        report,code=judge.evaluate(save(tmp_path,truth));assert code==2 and report['metrics']=={} and 'Evaluator' in report['errors'][0]
    finally:judge.ASSETS=previous
def test_bad_metric_configuration(tmp_path,truth):
    previous=judge.ASSETS
    try:
        judge.ASSETS=tmp_path;(tmp_path/'metrics.json').write_text('{"thresholds":{}}')
        report,code=judge.evaluate(save(tmp_path,truth));assert code==2 and report['metrics']=={}
    finally:judge.ASSETS=previous
def test_report_write_failure(tmp_path,truth):
    prediction=save(tmp_path,truth);block=tmp_path/'block';block.write_text('file')
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(block/'report.json')],cwd=tmp_path,capture_output=True)
    assert result.returncode==2 and b'Cannot write report' in result.stderr
def test_metadata_and_allowlist():
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert '| traction | (1,2,50,50) | float64 |' in text
    for token in ['github.com','github.io','githubusercontent.com','ground_truth.npz','baseline_reference.npz']:assert token not in text
    for token in ['2','50','100','kPa','micrometres']:assert token in text
