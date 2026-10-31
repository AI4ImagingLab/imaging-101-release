from pathlib import Path
import importlib.util,subprocess,sys,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
def ref():return j.load_arrays(ROOT/'evaluation/baseline_reference.npz')
def test_valid_and_poor(tmp_path):
    p=tmp_path/'v.npz';r=ref();np.savez_compressed(p,**r);assert j.evaluate(p)[1]==0
    np.savez_compressed(p,**{k:np.zeros_like(v) for k,v in r.items()});report,c=j.evaluate(p);assert c==1 and report['status']=='fail'
@pytest.mark.parametrize('kind',['missing_key','shape','dtype','complex','nan','inf'])
def test_malformed(tmp_path,kind):
    r=ref();key='reconstruction'
    if kind=='missing_key':del r[key]
    elif kind=='shape':r[key]=r[key][0]
    elif kind=='dtype':r[key]=r[key].astype(np.float32)
    elif kind=='complex':r[key]=r[key].astype(complex)
    else:r[key].flat[0]=np.nan if kind=='nan' else np.inf
    p=tmp_path/'bad.npz';np.savez_compressed(p,**r);report,c=j.evaluate(p)
    assert c==2 and report['metrics']=={} and report['errors']
def test_missing_and_evaluator_error(tmp_path,monkeypatch):
    assert j.evaluate(tmp_path/'none')[1]==2
    monkeypatch.setattr(j,'HERE',tmp_path);report,c=j.evaluate(ROOT/'output/reconstruction.npz');assert c==2 and report['metrics']=={}
def test_cli_foreign_directory_and_write_failure(tmp_path):
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    cmd[-1]=str(tmp_path);assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
def test_independent_metrics_case():
    r=ref();p={k:v.copy() for k,v in r.items()};p['reconstruction']+=.01
    m=j.metrics(p,r);assert np.isclose(m['fraction_max_error'],.01)
    assert m['cd_refit_rmse']==0 and m['ir_refit_rmse']==0
def test_undefined_configuration(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_text('{"thresholds":{}}',encoding='utf8')
    monkeypatch.setattr(j,'HERE',tmp_path)
    report,code=j.evaluate(ROOT/'output/reconstruction.npz')
    assert code==2 and 'metric' in report['errors'][0]
