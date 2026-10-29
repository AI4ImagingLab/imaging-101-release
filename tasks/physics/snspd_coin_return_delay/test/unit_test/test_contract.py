from pathlib import Path
import importlib.util,json,sys,subprocess,shutil
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('snspd_judge',ROOT/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
REF=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
from src.preprocessing import normalize_irf,load_data
from src.solvers import zero_norm,circular_correlation,reconstruct
from src.physics_model import expected_histogram,BIN_DEPTH_MM
def save(tmp,a):p=tmp/'pred.npz';np.savez(p,reconstruction=a);return p
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','object'])
def test_bad_inputs(tmp_path,kind):
    p=tmp_path/'bad.npz';a=REF.copy()
    if kind=='key':np.savez(p,wrong=a)
    elif kind!='missing':
        if kind=='shape':a=a[0]
        if kind=='dtype':a=a.astype(np.float32)
        if kind=='complex':a=a.astype(complex)
        if kind=='nan':a[0,0,0]=np.nan
        if kind=='inf':a[0,0,0]=np.inf
        if kind=='object':a=a.astype(object)
        np.savez(p,reconstruction=a)
    r,c=J.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
def test_zero_and_wrong_peak_fail(tmp_path):
    r,c=J.evaluate(save(tmp_path,np.zeros_like(REF)));assert c==1 and not r['passed']
    with np.load(ROOT/'test/fixtures/upstream.npz') as z:s=z['peaks'].copy()
    reference=s[0,0];s=np.minimum(s+20,9999);s[0,0]=reference;x=((s-reference)*BIN_DEPTH_MM)[None]
    r,c=J.evaluate(save(tmp_path,x));assert c==1 and not r['passed']
@pytest.mark.parametrize('kind',['fraction','gauge','window','huge'])
def test_invalid_timing_conventions(tmp_path,kind):
    a=REF.copy()
    if kind=='fraction':a[0,1,1]+=.1*BIN_DEPTH_MM
    if kind=='gauge':a+=BIN_DEPTH_MM
    if kind=='window':a[0,1,1]=1e5*BIN_DEPTH_MM
    if kind=='huge':a[0,1,1]=1e308
    r,c=J.evaluate(save(tmp_path,a));assert c==2 and r['metrics']=={}
@pytest.mark.parametrize('value',[-1,True,float('inf'),float('nan'),'x'])
def test_invalid_boundary(tmp_path,monkeypatch,value):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['mean_abs_bin_error_boundary']=value;(tmp_path/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(J,'ASSETS',tmp_path);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
def test_invalid_regret_range(tmp_path,monkeypatch):
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['mean_correlation_regret_boundary']=3.;(tmp_path/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(J,'ASSETS',tmp_path);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2
def test_known_metrics():
    a=np.array([0.,BIN_DEPTH_MM,2*BIN_DEPTH_MM]);r=J.score(a,np.zeros(3));assert r['max_abs_bin_error']==pytest.approx(2) and r['mean_abs_bin_error']==pytest.approx(1)
def test_truth_priority_without_fallback(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir()
    for f in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/f,assets/f)
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');monkeypatch.setattr(J,'ASSETS',assets);np.savez(assets/'ground_truth.npz',reconstruction=np.zeros_like(REF));r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==1
    np.savez(assets/'ground_truth.npz',wrong=REF);r,c=J.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
def test_foreign_cwd_and_report_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')];p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed'];cmd[-1]=str(tmp_path);p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Cannot write report' in p.stderr
@pytest.mark.parametrize('a',[np.zeros(8),np.array([-1.,2.]),np.array([np.nan,1.]),np.ones((1,4)),np.ones(10)*1e308])
def test_invalid_irf(a):
    with pytest.raises(ValueError):normalize_irf(a)
def test_flat_histogram_and_length_validation():
    np.testing.assert_array_equal(zero_norm(np.ones(20)),np.zeros(20))
    with pytest.raises(ValueError):circular_correlation(np.ones(20),np.ones(21))
    with pytest.raises(ValueError):reconstruct(np.ones((2,2,10)),np.ones(10))
    with pytest.raises(ValueError):expected_histogram(np.ones(10),.5,1,0)
def test_public_metadata_contract_and_no_urls():
    text=(ROOT/'README.md').read_text(encoding='utf-8');assert '(1,50,50)' in text and '0.149896229' in text and '299792458' in text and '1 | ps' in text
    for name in ['README.md','requirements.txt']:
        t=(ROOT/name).read_text(encoding='utf-8').lower();assert not any(k in t for k in ['github.com','github.io','githubusercontent','codeload','baseline_reference.npz','ground_truth.npz'])
def test_library_backend_unchanged():
    import matplotlib
    b=matplotlib.get_backend();import src.main,src.visualization
    assert b==matplotlib.get_backend()
