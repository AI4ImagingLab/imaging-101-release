from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_numeric_metric_and_visual_agreement():
    from src.visualization import metric_pair
    b=np.array([[0.,1.,2.]]);a=2*b;mask=np.array([[True,False,True]])
    m=judge.score(a,b,mask)
    assert m['ncc']==pytest.approx(1.)
    assert m['nrmse']==pytest.approx(np.sqrt(5/3)/2)
    assert m['missing_nrmse']==pytest.approx(.5)
    assert judge.score(np.zeros_like(a),b,mask)['ncc']==0
    v=metric_pair(a,b,mask)
    for k in ['ncc','nrmse','missing_nrmse']:assert v[k]==pytest.approx(m[k])
    with pytest.raises(ValueError):judge.score(a,np.ones_like(b),mask)

def test_true_and_fresh_outputs_and_poor_fills(tmp_path):
    for path in [ROOT/'evaluation/ground_truth.npz',ROOT/'output/reconstruction.npz']:
        report,code=judge.evaluate(path);assert code==0 and report['passed']
    for name,value in [('zero',0.),('constant',.5)]:
        path=tmp_path/f'{name}.npz';np.savez_compressed(path,reconstruction=np.full(judge.SHAPE,value))
        report,code=judge.evaluate(path);assert code==1 and report['status']=='fail'

@pytest.mark.parametrize('problem',['missing','key','shape','float32','complex','nan','inf','object','overflow'])
def test_invalid_prediction(tmp_path,problem):
    path=tmp_path/'x.npz';x=np.zeros(judge.SHAPE)
    if problem=='missing':pass
    elif problem=='key':np.savez(path,wrong=x)
    else:
        if problem=='shape':x=x[0]
        if problem=='float32':x=x.astype('float32')
        if problem=='complex':x=x.astype('complex128')
        if problem=='nan':x.flat[0]=np.nan
        if problem=='inf':x.flat[0]=np.inf
        if problem=='object':x=np.array(['hidden'],dtype=object)
        if problem=='overflow':x[:]=1e308
        np.savez(path,reconstruction=x)
    report,code=judge.evaluate(path)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('key,value',[('ncc_boundary',float('nan')),('ncc_boundary',float('inf')),('ncc_boundary',True),('ncc_boundary',1.1),('nrmse_boundary',-1.),('nrmse_boundary','0.2'),('missing_nrmse_boundary',-1.),('known_pixel_max_error_boundary',None)])
def test_bad_configuration(tmp_path,monkeypatch,key,value):
    assets=tmp_path/'evaluation';assets.mkdir();cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg[key]=value
    (assets/'metrics.json').write_text(json.dumps(cfg),encoding='utf8');monkeypatch.setattr(judge,'ASSETS',assets)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==2 and report['metrics']=={}
    assert 'boundary' in report['errors'][0] or 'numeric' in report['errors'][0]

def test_missing_assets_configuration_and_degenerate_reference(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();monkeypatch.setattr(judge,'ASSETS',assets)
    report,code=judge.evaluate('missing');assert code==2
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg.pop('ncc_boundary')
    (assets/'metrics.json').write_text(json.dumps(cfg));report,code=judge.evaluate('missing');assert code==2
    shutil.copyfile(ROOT/'evaluation/metrics.json',assets/'metrics.json');report,code=judge.evaluate('missing');assert code==2
    np.savez_compressed(assets/'ground_truth.npz',reconstruction=np.ones(judge.SHAPE))
    (tmp_path/'data').mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==2 and report['metrics']=={}

def test_boundaries_actually_consumed(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'evaluation/ground_truth.npz',assets/'ground_truth.npz');shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['nrmse_boundary']=0.
    (assets/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(judge,'ASSETS',assets)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==1 and report['metrics']['nrmse']>0

def test_foreign_cli_and_report_write_failure(tmp_path):
    target=tmp_path/'nested/report.json';command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(target)]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True);assert p.returncode==0 and json.loads(target.read_text())['passed']
    p=subprocess.run(command[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'write report' in p.stderr

def test_public_metadata_and_native_output_parity():
    from src.preprocessing import load_observations
    y,mask=load_observations()
    assert y.shape==(240,320) and mask.dtype==bool and mask.mean()==pytest.approx(.3016536458333333)
    with np.load(ROOT/'notebooks/reference_outputs/upstream_methods.npz',allow_pickle=False) as z, np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as p:
        np.testing.assert_allclose(p['reconstruction'][0],z['order2'],rtol=0,atol=1e-10)
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['240','320','0.3016536458333333','0.3R+0.59G+0.11B']:assert value in text

