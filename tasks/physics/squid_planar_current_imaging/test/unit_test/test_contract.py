from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return f['reconstruction'].copy()
def test_known_metrics_and_independent_forward():
    assert j.cosine(np.array([0.,2.]),np.array([0.,1.]))==1
    assert j.relative_l2(np.array([0.,2.]),np.array([0.,1.]))==1
    assert j.cosine(np.zeros(2),np.array([0.,1.]))==0
    with pytest.raises(ValueError):j.relative_l2(np.ones(2),np.zeros(2))
    r,c=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==0,r
@pytest.mark.parametrize('kind',['zero','scale','offset','swap','missing','key','shape','dtype','complex','nan','inf','object','overflow'])
def test_strict_bad_input_and_quality(tmp_path,kind):
    x=baseline();key='reconstruction'
    if kind=='zero':x*=0
    if kind=='scale':x*=2
    if kind=='offset':x+=x.max()*.1
    if kind=='swap':x=x[:,::-1].copy()
    if kind=='key':key='wrong'
    if kind=='shape':x=x[:,:,:151,:201]
    if kind=='dtype':x=x.astype('float32')
    if kind=='complex':x=x.astype('complex128')
    if kind=='nan':x.flat[0]=np.nan
    if kind=='inf':x.flat[0]=np.inf
    if kind=='object':x=np.array([object()],dtype=object)
    if kind=='overflow':x.fill(np.finfo(float).max)
    path=tmp_path/'prediction.npz'
    if kind!='missing':np.savez_compressed(path,**{key:x})
    r,c=j.evaluate(path);assert c==(1 if kind in ['zero','scale','offset','swap'] else 2),(kind,r)
    if c==2:assert r['metrics']=={} and r['errors']
def test_evaluator_failure_and_actual_truth_precedence(tmp_path,monkeypatch):
    ev=tmp_path/'evaluation';ev.mkdir();(tmp_path/'data').mkdir()
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy(ROOT/'evaluation'/name,ev/name)
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    np.savez_compressed(ev/'ground_truth.npz',reconstruction=np.zeros((1,2,301,401)))
    monkeypatch.setattr(j,'ROOT',ev);r,c=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==2 and r['metrics']=={}
    monkeypatch.setattr(j,'ROOT',tmp_path/'absent');r,c=j.evaluate('absent');assert c==2 and r['errors']
def test_top_level_canonical_boundaries_are_consumed(tmp_path,monkeypatch):
    ev=tmp_path/'evaluation';ev.mkdir();(tmp_path/'data').mkdir()
    shutil.copy(ROOT/'evaluation/baseline_reference.npz',ev/'baseline_reference.npz')
    shutil.copy(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    config=json.loads((ROOT/'evaluation/metrics.json').read_text())
    assert 'thresholds' not in config
    config['ncc_jx_boundary']=1.1
    (ev/'metrics.json').write_text(json.dumps(config))
    monkeypatch.setattr(j,'ROOT',ev);r,c=j.evaluate(ROOT/'evaluation/baseline_reference.npz')
    assert c==1 and r['thresholds']['ncc_jx']['boundary']==1.1
    del config['ncc_jx_boundary'];(ev/'metrics.json').write_text(json.dumps(config))
    r,c=j.evaluate(ROOT/'evaluation/baseline_reference.npz');assert c==2 and r['metrics']=={}
def test_report_write_failure(tmp_path):
    (tmp_path/'block').write_text('file');cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'evaluation/baseline_reference.npz'),'--report',str(tmp_path/'block/report.json')]
    p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==2 and 'Cannot write' in p.stderr
def test_public_arrays_and_acquisition_metadata():
    text=(ROOT/'README.md').read_text(encoding='utf8');assert 'github' not in text.lower()
    expected={'flux':(1,151,201),'xs':(1,201),'ys':(1,151),'psf_native':(1,209,209),'psf_xs':(1,209),'psf_ys':(1,209),'psf_coarse':(1,301,401),'psf_coarse_xs':(1,401),'psf_coarse_ys':(1,301)}
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        assert set(f.files)==set(expected)
        for key,shape in expected.items():assert f[key].shape==shape and f[key].dtype==np.float64 and '| '+key+' | '+str(shape)+' | float64 |' in text
    for value in ['1.2924','1.144','1.295272','1.147336666','301','401','linear_ramp','zero','600']:assert value in text
