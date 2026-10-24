from pathlib import Path
import importlib.util,numpy as np,json,pytest,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def baseline():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return {k:f[k] for k in f.files}
def test_independent_metrics_known_case():
    values=judge.numeric_metrics(np.array([1.,2.,3.]),np.array([2.,4.,6.]))
    assert values['ncc']==pytest.approx(1);assert values['nrmse']==pytest.approx(np.sqrt(14/3)/4)
@pytest.mark.parametrize('mode',['missing','key','shape','dtype','complex','nan','inf'])
def test_bad_submission_errors(tmp_path,mode):
    p=tmp_path/'pred.npz';arrays=baseline()
    if mode=='key':arrays.pop('fat')
    if mode=='shape':arrays['water']=arrays['water'][...,:-1]
    if mode=='dtype':arrays['water']=arrays['water'].astype(np.float64)
    if mode=='complex':arrays['water']=arrays['water'].astype(np.complex64)
    if mode=='nan':arrays['fat'].flat[0]=np.nan
    if mode=='inf':arrays['r2star'].flat[0]=np.inf
    if mode!='missing':np.savez(p,**arrays)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('key',list(judge.DTYPES))
def test_each_required_quantity_rejects_wrong_output(tmp_path,key):
    arrays=baseline();arrays[key]*=0;p=tmp_path/'poor.npz';np.savez(p,**arrays);r,c=judge.evaluate(p)
    assert c==1 and r['metrics'][key+'_ncc']==0
def test_schema_assets_priority_boundaries_and_foreign_cwd(tmp_path,monkeypatch):
    evaluation=tmp_path/'evaluation';evaluation.mkdir();arrays=baseline();p=tmp_path/'p.npz';np.savez(p,**arrays)
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text());(evaluation/'metrics.json').write_text(json.dumps(cfg));monkeypatch.setattr(judge,'ROOT',evaluation)
    r,c=judge.evaluate(p);assert c==2 and not r['metrics']
    np.savez(evaluation/'baseline_reference.npz',wrong=np.zeros(1));r,c=judge.evaluate(p);assert c==2
    np.savez(evaluation/'ground_truth.npz',**arrays);r,c=judge.evaluate(p);assert c==0
    arrays['water']*=1.00001;np.savez(p,**arrays);r,c=judge.evaluate(p);score=r['metrics']['water_nrmse'];assert 0<score<1e-5
    for factor,exitcode in [(.99,1),(1.01,0)]:
        cfg['water_nrmse_boundary']=score*factor;(evaluation/'metrics.json').write_text(json.dumps(cfg));r,c=judge.evaluate(p);assert c==exitcode
    cfg['water_nrmse_boundary']=float('nan');(evaluation/'metrics.json').write_text(json.dumps(cfg));r,c=judge.evaluate(p);assert c==2 and not r['metrics']
def test_cli_and_report_write_failure(tmp_path):
    report=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(tmp_path/'none'),'--report',str(tmp_path)],capture_output=True,text=True);assert result.returncode==2 and 'Cannot write report' in result.stderr

@pytest.mark.parametrize('mode',['shape','dtype','nan','constant'])
def test_malformed_reference_errors(tmp_path,monkeypatch,mode):
    evaluation=tmp_path/'evaluation';evaluation.mkdir();arrays=baseline();p=tmp_path/'p.npz';np.savez(p,**arrays)
    if mode=='shape':arrays['water']=arrays['water'][...,:-1]
    if mode=='dtype':arrays['water']=arrays['water'].astype(np.float64)
    if mode=='nan':arrays['fieldmap'].flat[0]=np.nan
    if mode=='constant':arrays['fat']*=0
    np.savez(evaluation/'baseline_reference.npz',**arrays)
    shutil.copyfile(ROOT/'evaluation/metrics.json',evaluation/'metrics.json');monkeypatch.setattr(judge,'ROOT',evaluation)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

@pytest.mark.parametrize('value',[True,None,-1.,float('inf'),'missing'])
def test_invalid_threshold_errors(tmp_path,monkeypatch,value):
    evaluation=tmp_path/'evaluation';evaluation.mkdir();shutil.copyfile(ROOT/'evaluation/baseline_reference.npz',evaluation/'baseline_reference.npz')
    cfg=json.loads((ROOT/'evaluation/metrics.json').read_text())
    if value=='missing':cfg.pop('water_nrmse_boundary')
    else:cfg['water_nrmse_boundary']=value
    (evaluation/'metrics.json').write_text(json.dumps(cfg));p=tmp_path/'p.npz';np.savez(p,**baseline());monkeypatch.setattr(judge,'ROOT',evaluation)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}

def test_extreme_finite_overflow_is_error(tmp_path):
    arrays=baseline();arrays['fieldmap'].fill(1e308);p=tmp_path/'huge.npz';np.savez(p,**arrays)
    r,c=judge.evaluate(p);assert c==2 and r['metrics']=={}
