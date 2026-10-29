from pathlib import Path
import importlib.util,sys,json,subprocess,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
@pytest.fixture
def submission():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in judge.KEYS}
def test_known_metrics():
    x=np.array([1.,2.,3.]);r=np.array([2.,2.,2.])
    with pytest.raises(ValueError):judge.score(x,r)
    ncc,err=judge.score(x,np.array([1.,1.,2.]))
    assert ncc==pytest.approx(9/np.sqrt(84));assert err==pytest.approx(np.sqrt(2/3))
@pytest.mark.parametrize('key',judge.KEYS)
def test_each_quantity_is_required_and_evaluated(tmp_path,submission,key):
    submission[key]*=0
    path=tmp_path/'prediction.npz';np.savez_compressed(path,**submission)
    report,code=judge.evaluate(path);assert code==1 and report['status']=='fail'
@pytest.mark.parametrize('kind',['missing','shape','float32','complex','nan','inf','overflow','object'])
def test_invalid_predictions(tmp_path,submission,kind):
    if kind=='missing':submission.pop('phase_E')
    elif kind=='shape':submission['phase_E']=submission['phase_E'][0]
    elif kind=='float32':submission['phase_E']=submission['phase_E'].astype(np.float32)
    elif kind=='complex':submission['phase_E']=submission['phase_E'].astype(complex)
    elif kind=='object':submission['phase_E']=np.array([object()],dtype=object)
    else:submission['phase_E'].fill({'nan':np.nan,'inf':np.inf,'overflow':1e308}[kind])
    path=tmp_path/'bad.npz';np.savez_compressed(path,**submission)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)
@pytest.mark.parametrize('boundary',[True,float('nan'),float('inf'),-1.1,1.1])
def test_invalid_thresholds(tmp_path,monkeypatch,boundary):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['phase_B_ncc_boundary']=boundary
    (tmp_path/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(judge,'ASSETS',tmp_path)
    report,code=judge.evaluate('absent.npz');assert code==2 and not report['metrics']
def test_actual_truth_priority_then_baseline(tmp_path,submission,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir()
    shutil.copy2(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg.update(observation_B_boundary=10.,observation_E_boundary=10.,induction_consistency_boundary=10.)
    (assets/'metrics.json').write_text(json.dumps(cfg))
    np.savez_compressed(assets/'baseline_reference.npz',**submission)
    truth={k:v*2 for k,v in submission.items()};np.savez_compressed(assets/'ground_truth.npz',**truth)
    pred=tmp_path/'prediction.npz';np.savez_compressed(pred,**truth);monkeypatch.setattr(judge,'ASSETS',assets)
    assert judge.evaluate(pred)[1]==0
    (assets/'ground_truth.npz').unlink();assert judge.evaluate(pred)[1]==1
    np.savez_compressed(pred,**submission);assert judge.evaluate(pred)[1]==0
    np.savez_compressed(assets/'ground_truth.npz',broken=np.ones(2));assert judge.evaluate(pred)[1]==2
def test_foreign_cwd_cli_and_missing_file(tmp_path):
    out=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(out)],cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads(out.read_text())['passed']
    report,code=judge.evaluate(tmp_path/'absent.npz');assert code==2 and report['metrics']=={}
def test_pipeline_entry_point(tmp_path):
    p=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True)
    assert p.returncode==0
    assert judge.evaluate(ROOT/'output/reconstruction.npz')[1]==0
def test_report_write_failure(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path)],capture_output=True)
    assert p.returncode==2 and b'Cannot write report' in p.stderr
def test_readme_acquisition_matches_constants():
    sys.path.insert(0,str(ROOT))
    from src.preprocessing import PIXEL_NM,VOLTAGE_V,DEFOCUS_NM
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert str(PIXEL_NM) in text and str(int(VOLTAGE_V)) in text
    for v in DEFOCUS_NM:assert str(int(v)) in text
