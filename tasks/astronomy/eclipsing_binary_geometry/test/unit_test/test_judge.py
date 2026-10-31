from pathlib import Path
import importlib.util,json,shutil,numpy as np,pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def point():
    with np.load(ROOT/'evaluation/baseline_reference.npz') as f:return f['reconstruction'].copy()

def run(tmp_path,arr=None,**keys):
    path=tmp_path/'prediction.npz';np.savez(path,**({'reconstruction':arr} if arr is not None else keys));report=tmp_path/'nested/report.json'
    code=judge.evaluate(path,report);return code,json.loads(report.read_text(encoding='utf8'))

def test_valid_original_and_diagnostic_sum(tmp_path):
    code,r=run(tmp_path,point());assert code==0 and r['passed']
    assert abs(sum(v for k,v in r['metrics'].items() if k!='total_chi_squared')-r['metrics']['total_chi_squared'])<1e-8

@pytest.mark.parametrize('i,delta',[(0,.01),(1,.003),(2,-2),(3,.05),(4,.03),(5,.2),(6,1.)])
def test_every_unknown_is_scored(tmp_path,i,delta):
    p=point();p[0,i]+=delta;code,r=run(tmp_path,p)
    assert code==1 and r['status']=='fail' and r['metrics']['total_chi_squared']>15900.910369432995

@pytest.mark.parametrize('kind',['missing','shape','integer','complex','nan','inf','extra','physical'])
def test_malformed_outputs(tmp_path,kind):
    p=point()
    if kind=='missing':code,r=run(tmp_path,wrong=p)
    elif kind=='extra':code,r=run(tmp_path,reconstruction=p,extra=np.ones(1))
    else:
        if kind=='shape':p=p.reshape(7)
        if kind=='integer':p=p.astype(int)
        if kind=='complex':p=p.astype(complex)
        if kind=='nan':p[0,0]=np.nan
        if kind=='inf':p[0,0]=np.inf
        if kind=='physical':p[0,0]=-1
        code,r=run(tmp_path,p)
    assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('kind',['empty','missing_metric','comparison','nan','negative','zero','wrong_fields','missing_config','missing_raw'])
def test_evaluator_assets_and_configuration(tmp_path,monkeypatch,kind):
    root=tmp_path/'task';(root/'evaluation').mkdir(parents=True);(root/'data').mkdir()
    shutil.copy2(ROOT/'data/raw_data.npz',root/'data/raw_data.npz');cfg=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    if kind=='empty':cfg['thresholds']={}
    if kind=='missing_metric':cfg['thresholds']={'wrong':{'comparison':'<=','boundary':1}}
    if kind=='comparison':cfg['thresholds']['total_chi_squared']['comparison']='>='
    if kind=='nan':cfg['thresholds']['total_chi_squared']['boundary']=float('nan')
    if kind=='negative':cfg['thresholds']['total_chi_squared']['boundary']=-1
    if kind=='zero':cfg['thresholds']['total_chi_squared']['boundary']=0
    if kind=='wrong_fields':cfg['thresholds']['total_chi_squared']['unused']=1
    if kind!='missing_config':(root/'evaluation/metrics.json').write_text(json.dumps(cfg),encoding='utf8')
    if kind=='missing_raw':(root/'data/raw_data.npz').unlink()
    monkeypatch.setattr(judge,'__file__',str(root/'evaluation/judge.py'))
    code,r=run(tmp_path,point());assert code==2 and r['metrics']=={} and r['errors']

def test_missing_file_foreign_cwd_and_report_write_failure(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);report=tmp_path/'missing.json'
    assert judge.evaluate(tmp_path/'absent.npz',report)==2
    code,r=run(tmp_path,point());assert code==0
    pred=tmp_path/'prediction.npz';block=tmp_path/'file';block.write_text('not a directory')
    assert judge.evaluate(pred,block/'report.json')==2
