from pathlib import Path
import sys,importlib.util,json,shutil,subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,normalize_models
from src.physics_model import highpass,calzetti,forward,adjoint,smooth_curve
from src.solvers import reconstruct,fit_population,sigma_clip
spec=importlib.util.spec_from_file_location('taskjudge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def assets(tmp_path):
    folder=tmp_path/'evaluation';shutil.copytree(ROOT/'evaluation',folder);shutil.copytree(ROOT/'data',tmp_path/'data')
    with np.load(folder/'baseline_reference.npz') as z:out={k:z[k].copy() for k in z.files}
    return folder,out,tmp_path/'prediction.npz'
def test_native_float32_and_known_metrics(assets):
    folder,out,path=assets
    for dtype in ['float64','float32']:
        np.savez(path,**{k:v.astype(dtype) for k,v in out.items()});report,code=judge.evaluate(path,folder);assert code==0,report
    with np.load(ROOT/'data/raw_data.npz') as z:data={k:z[k] for k in z.files}
    pred={k:v.copy() for k,v in out.items()};pred['ebv']+=.02
    assert judge.score(pred,out,data)['ebv_absolute_error']==pytest.approx(.02)
    pred['attenuation']+=.001
    assert judge.score(pred,out,data)['attenuation_nrmse']==pytest.approx(.001/np.ptp(out['attenuation']))
@pytest.mark.parametrize('bad',['missing','shape','integer','complex','nan','inf','negative','sum','attenuation','ebv','object'])
def test_invalid_inputs(assets,bad):
    folder,out,path=assets
    if bad=='missing':out.pop('ebv')
    elif bad=='shape':out['reconstruction']=out['reconstruction'][0]
    elif bad in ['integer','complex','object']:out['reconstruction']=out['reconstruction'].astype({'integer':'int64','complex':'complex128','object':'object'}[bad])
    elif bad in ['nan','inf']:out['ebv'][0,0]=float(bad)
    elif bad=='negative':out['reconstruction'][0,0]=-1
    elif bad=='sum':out['reconstruction']*=2
    elif bad=='attenuation':out['attenuation'][0,0]=0
    elif bad=='ebv':out['ebv'][0,0]=2
    np.savez(path,**out);report,code=judge.evaluate(path,folder);assert code==2 and report['metrics']=={} and report['errors']
def test_missing_and_poor_valid_output(assets):
    folder,out,path=assets;assert judge.evaluate(path,folder)[1]==2
    out['reconstruction'][:]=1/198;np.savez(path,**out);report,code=judge.evaluate(path,folder);assert code==1 and report['status']=='fail'
def test_v2_has_no_hidden_population_dependency(assets):
    folder,out,path=assets;np.savez(path,**out);np.savez(folder/'ground_truth.npz',**out)
    changed={k:v.copy() for k,v in out.items()};changed['ebv']+=.1;np.savez(folder/'baseline_reference.npz',**changed)
    assert judge.evaluate(path,folder)[1]==0
    changed['reconstruction']*=2;np.savez(folder/'ground_truth.npz',**changed);assert judge.evaluate(path,folder)[1]==0
def test_assets_config_and_report_failure(assets):
    folder,out,path=assets;np.savez(path,**out)
    config=json.loads((folder/'metrics.json').read_text());config['log_attenuation_tv_boundary']=float('nan');(folder/'metrics.json').write_text(json.dumps(config));assert judge.evaluate(path,folder)[1]==2
    (folder/'metrics.json').unlink();assert judge.evaluate(path,folder)[1]==2
    assert judge.main(['--prediction',str(path),'--report',str(path/'impossible.json')])==2
def test_external_cwd_cli_and_json(assets):
    folder,out,path=assets;np.savez(path,**out);report=path.parent/'nested/report.json'
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(path),'--report',str(report)],cwd=path.parent,capture_output=True,text=True)
    assert r.returncode==0,r.stderr;assert json.loads(report.read_text())['passed']


def test_positive_grey_is_physical_equivalence(assets):
    folder,out,path=assets
    np.savez(path,**out);before,code=judge.evaluate(path,folder);assert code==0
    out['attenuation']*=7;np.savez(path,**out);after,code=judge.evaluate(path,folder);assert code==0
    for k in before['metrics']:assert after['metrics'][k]==pytest.approx(before['metrics'][k],abs=1e-12)
