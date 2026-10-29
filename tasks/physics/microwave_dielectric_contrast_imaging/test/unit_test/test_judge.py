from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('microwave_judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def known():
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz',allow_pickle=False) as f:return f['reconstruction'].copy()
def test_metrics_independently_known_cases():
    assert j.ncc(np.array([0.,2.]),np.array([0.,1.]))==1
    assert j.ncc(np.zeros(2),np.array([0.,1.]))==0
    assert j.nrmse(np.array([1.,3.]),np.array([0.,2.]))==.5
    with pytest.raises(ValueError):j.nrmse(np.ones(2),np.ones(2))
    with pytest.raises(ValueError):j.ncc(np.ones(2),np.zeros(2))
@pytest.mark.parametrize('kind',['zero','scale','missing','key','shape','dtype','complex','nan','inf','object','overflow'])
def test_strict_contract_and_poor_predictions(tmp_path,kind):
    x=known();key='reconstruction'
    if kind=='zero':x*=0
    if kind=='scale':x*=2
    if kind=='key':key='wrong'
    if kind=='shape':x=x[:,:,:3]
    if kind=='dtype':x=x.astype('float32')
    if kind=='complex':x=x.astype('complex128')
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='object':x=np.array([object()],dtype=object)
    if kind=='overflow':x.fill(np.finfo(float).max)
    pred=tmp_path/'prediction.npz'
    if kind!='missing':np.savez_compressed(pred,**{key:x})
    report,code=j.evaluate(pred)
    expected=1 if kind in ['zero','scale'] else 2
    assert code==expected,(kind,report)
    if code==2:assert report['metrics']=={} and report['errors']
def test_actual_truth_and_original_baseline_both_pass():
    for path in [ROOT/'evaluation/ground_truth.npz',ROOT/'notebooks/reference_outputs/upstream.npz']:
        report,code=j.evaluate(path);assert code==0,report
def test_foreign_cwd_report_creation_and_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'notebooks/reference_outputs/upstream.npz'),'--report',str(tmp_path/'new/report.json')]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'new/report.json').read_text());assert report['passed'] and not report['errors']
    (tmp_path/'block').write_text('file');command[-1]=str(tmp_path/'block/report.json');result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==2 and 'Cannot write' in result.stderr
def test_truth_precedence_and_evaluator_failure(tmp_path,monkeypatch):
    ev=tmp_path/'evaluation';ev.mkdir();(tmp_path/'data').mkdir()
    for file in ['ground_truth.npz','metrics.json']:shutil.copy(ROOT/'evaluation'/file,ev/file)
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    np.savez_compressed(ev/'baseline_reference.npz',reconstruction=np.zeros((1,30,30)))
    monkeypatch.setattr(j,'ROOT',ev);report,code=j.evaluate(ROOT/'notebooks/reference_outputs/upstream.npz');assert code==0,report
    monkeypatch.setattr(j,'ROOT',tmp_path/'missing');report,code=j.evaluate('missing.npz');assert code==2 and report['metrics']=={} and report['errors']
def test_independent_dense_physical_forward_matches_frozen_field():
    with np.load(ROOT/'notebooks/reference_outputs/upstream.npz') as f:x=f['reconstruction'][0]
    with np.load(ROOT/'data/raw_data.npz') as f:d={k:f[k][0] for k in f.files}
    with np.load(ROOT/'test/fixtures/oracle_stages.npz') as f:
        expected=f['output_GS']@(x.ravel()[:,None]*f['output_final_total_field'])
    np.testing.assert_allclose(j.physical_prediction(x,d['illumination_angles'],d['receiver_angles']),expected,rtol=2e-7,atol=2e-8)
