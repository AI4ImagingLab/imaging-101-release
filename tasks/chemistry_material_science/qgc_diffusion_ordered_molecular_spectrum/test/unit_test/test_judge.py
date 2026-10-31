from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_independent_metrics_known_case():
    ref=np.ones((29,256));pred=ref*2;K=np.ones((2,256));Y=np.ones((29,2))*256;eta=np.ones(29)*256*np.sqrt(2)
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    metrics,_=judge.score(pred,ref,K,Y,eta,cfg)
    assert metrics['maximum_band_relative_l2']==pytest.approx(1.)
    assert metrics['minimum_band_cosine']==pytest.approx(1.)
    assert metrics['maximum_observation_ball_ratio']==pytest.approx(1.)
    assert metrics['maximum_negative_relative_amplitude']==0
    metrics,_=judge.score(-ref,ref,K,Y,eta,cfg)
    assert metrics['minimum_band_cosine']==pytest.approx(-1.)
    assert metrics['maximum_negative_relative_amplitude']==pytest.approx(1.)

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','poor'])
def test_malformed_and_poor(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:x=f['reconstruction']
    if kind=='shape':x=x[0]
    if kind=='dtype':x=x.astype('float32')
    if kind=='complex':x=x.astype('complex128')
    if kind in ['nan','inf']:x.flat[0]=float(kind)
    if kind=='poor':x*=.5
    pred=tmp_path/'p.npz';report=tmp_path/'sub/r.json'
    if kind!='missing':np.savez(pred,**{('wrong' if kind=='key' else 'reconstruction'):x})
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path,capture_output=True)
    r=json.loads(report.read_text());assert result.returncode==(1 if kind=='poor' else 2)
    assert not r['passed']
    if kind!='poor':assert r['metrics']=={} and r['errors']

def test_assets_precedence_and_write_failure(tmp_path):
    shutil.copytree(ROOT/'evaluation',tmp_path/'evaluation');shutil.copytree(ROOT/'data',tmp_path/'data')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:x=f['reconstruction']
    np.savez(tmp_path/'p.npz',reconstruction=x);np.savez(tmp_path/'evaluation/ground_truth.npz',reconstruction=x*2)
    cmd=[sys.executable,str(tmp_path/'evaluation/judge.py'),'--prediction',str(tmp_path/'p.npz'),'--report',str(tmp_path/'r.json')]
    assert subprocess.run(cmd,capture_output=True).returncode==0
    (tmp_path/'evaluation/ground_truth.npz').unlink();(tmp_path/'evaluation/baseline_reference.npz').unlink()
    assert subprocess.run(cmd,capture_output=True).returncode==0
    result=subprocess.run(cmd[:-1]+[str(tmp_path)],capture_output=True)
    assert result.returncode==2 and b'Report-write failure' in result.stderr


def test_native_finite_prediction_metric_overflow_writes_error_json(tmp_path):
    """Finite valid-shape inputs must not prevent a machine-readable report."""
    arrays={'reconstruction':np.full((1,29,256),1e300,dtype=np.float64)}
    assert all(np.isfinite(value).all() for value in arrays.values())
    prediction=tmp_path/'finite_huge.npz';report=tmp_path/'nested/error.json'
    np.savez_compressed(prediction,**arrays)
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True)
    assert result.returncode==2,result.stderr
    assert report.is_file()
    def reject_nonfinite(value):raise AssertionError('Nonfinite JSON number: '+value)
    record=json.loads(report.read_text(encoding='utf8'),parse_constant=reject_nonfinite)
    assert record['status']=='error' and record['passed'] is False
    assert record['metrics']=={} and record['thresholds']=={}
    assert record['errors'] and 'Nonfinite metric result' in record['errors'][0]
