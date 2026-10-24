from pathlib import Path
import importlib.util,numpy as np,pytest,json,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return f['reconstruction']

@pytest.mark.parametrize('mode',['missing','key','shape','dtype','complex','nan','inf'])
def test_malformed_predictions(tmp_path,mode):
    p=tmp_path/'bad.npz';a=np.ones(judge.SHAPE)
    if mode=='shape':a=a[...,:-1]
    if mode=='dtype':a=a.astype(np.float32)
    if mode=='complex':a=a.astype(np.complex128)
    if mode=='nan':a.flat[0]=np.nan
    if mode=='inf':a.flat[0]=np.inf
    if mode!='missing':np.savez(p,**({'wrong':a} if mode=='key' else {'reconstruction':a}))
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('mode',['zero','constant','sphere','single_voxel','checkerboard','high_frequency_mixture'])
def test_obvious_poor_spatial_outputs_fail(tmp_path,mode):
    shape=(64,64,64);x=np.indices(shape)-32;radius=np.sqrt(np.sum(x*x,axis=0))*3.75
    if mode=='zero':a=np.zeros(shape)
    if mode=='constant':a=np.ones(shape)
    if mode=='sphere':a=(radius<20).astype(float)
    if mode=='single_voxel':a=(radius<1).astype(float)
    if mode=='checkerboard':a=(np.indices(shape).sum(axis=0)%2).astype(float)
    if mode=='high_frequency_mixture':a=baseline()[0]*(1+.2*(-1.)**np.indices(shape).sum(axis=0))
    if a.sum():a=a/(a.sum()*3.75**3)
    p=tmp_path/'poor.npz';np.savez(p,reconstruction=a[None]);r,c=judge.evaluate(p)
    assert c==1 and not r['passed'],r

@pytest.mark.parametrize('key,value',[('chi2_boundary',float('nan')),('chi2_boundary',float('inf')),('chi2_boundary',True),('chi2_boundary',None),('mass_error_boundary',-1),('negative_mass_fraction_boundary',1.1),('rg_relative_error_boundary','missing')])
def test_invalid_metric_config(tmp_path,monkeypatch,key,value):
    evaluator=tmp_path/'evaluation';evaluator.mkdir();data=tmp_path/'data';data.mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',data/'raw_data.npz');cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if value=='missing':cfg.pop(key)
    else:cfg[key]=value
    (evaluator/'metrics.json').write_text(json.dumps(cfg));p=tmp_path/'pred.npz';np.savez(p,reconstruction=baseline());monkeypatch.setattr(judge,'ROOT',evaluator)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_extreme_finite_overflow_rejected(tmp_path):
    p=tmp_path/'extreme.npz';np.savez(p,reconstruction=np.full(judge.SHAPE,1e308));r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_all_independent_original_seeds_pass_without_spatial_reference(tmp_path):
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:
        for key in ['reconstruction','density_seed42','density_seed43','density_seed44','density_seed45']:
            p=tmp_path/'original.npz';np.savez(p,reconstruction=f[key]);r,c=judge.evaluate(p);assert c==0,r

def test_independent_judge_rotation_hand_translation_invariance(tmp_path):
    a=baseline();p=tmp_path/'transformed.npz'
    for b in [np.rot90(a,axes=(1,3)),a[:,:,::-1,:],np.roll(a,7,axis=2)]:
        np.savez(p,reconstruction=b);r,c=judge.evaluate(p);assert c==0,r

def test_tightened_threshold_is_enforced(tmp_path,monkeypatch):
    evaluator=tmp_path/'evaluation';evaluator.mkdir();(tmp_path/'data').mkdir();shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz');shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',evaluator/'baseline_reference.npz');cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());cfg['chi2_boundary']=0.;(evaluator/'metrics.json').write_text(json.dumps(cfg));p=tmp_path/'pred.npz';np.savez(p,reconstruction=baseline());monkeypatch.setattr(judge,'ROOT',evaluator);r,c=judge.evaluate(p);assert c==1 and r['metrics']['chi2']>0

@pytest.mark.parametrize('mode',['missing','key','shape','dtype','nan','constant'])
def test_reference_schema_errors(tmp_path,monkeypatch,mode):
    evaluator=tmp_path/'evaluation';evaluator.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    shutil.copyfile(ROOT/'evaluation/metrics.json',evaluator/'metrics.json')
    a=baseline();p=tmp_path/'prediction.npz';np.savez(p,reconstruction=a)
    b=a.copy()
    if mode=='shape':b=b[...,:-1]
    if mode=='dtype':b=b.astype(np.float32)
    if mode=='nan':b.flat[0]=np.nan
    if mode=='constant':b=np.zeros_like(b)
    if mode!='missing':np.savez(evaluator/'baseline_reference.npz',**({'wrong':b} if mode=='key' else {'reconstruction':b}))
    monkeypatch.setattr(judge,'ROOT',evaluator);r,c=judge.evaluate(p)
    assert c==2 and r['metrics']=={} and r['errors']

def test_actual_truth_precedence_and_reference_boundary_consumed(tmp_path,monkeypatch):
    evaluator=tmp_path/'evaluation';evaluator.mkdir();(tmp_path/'data').mkdir()
    shutil.copyfile(ROOT/'data/raw_data.npz',tmp_path/'data/raw_data.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    a=baseline();np.savez(evaluator/'ground_truth.npz',reconstruction=a)
    # A malformed lower-priority asset must be ignored when actual truth exists.
    np.savez(evaluator/'baseline_reference.npz',wrong=np.zeros(1))
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=a*(1+1e-7))
    monkeypatch.setattr(judge,'ROOT',evaluator)
    boundary=cfg['reference_scattering_nrmse_boundary']
    data={k:v for k,v in np.load(ROOT/'data/raw_data.npz').items()}
    measured=judge.reference_scattering_nrmse((a*(1+1e-7))[0],a[0],data)
    assert 0<measured<boundary
    for factor,expected in [(1.01,0),(.99,1)]:
        cfg['reference_scattering_nrmse_boundary']=factor*measured
        (evaluator/'metrics.json').write_text(json.dumps(cfg))
        report,code=judge.evaluate(p);assert code==expected,report
        assert report['metrics']['reference_scattering_nrmse']==pytest.approx(measured)

def test_missing_evaluator_asset(tmp_path,monkeypatch):
    p=tmp_path/'pred.npz';np.savez(p,reconstruction=baseline());monkeypatch.setattr(judge,'ROOT',tmp_path);r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_report_write_failure(tmp_path):
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'missing'),'--report',str(tmp_path)],capture_output=True,text=True);assert r.returncode==2 and 'Cannot write report' in r.stderr
