from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def test_independent_known_metrics():
    r=np.array([0.,1.,2.]);x=np.array([0.,2.,4.]);m=judge.image_metrics(x,r)
    assert abs(m['ncc']-1)<1e-15;np.testing.assert_allclose(m['nrmse'],np.sqrt(5/3)/2)
    assert judge.image_metrics(np.zeros(3),r)['ncc']==0
    with pytest.raises(ValueError):judge.image_metrics(x,np.ones(3))


def test_good_and_physically_wrong_output(tmp_path):
    good=ROOT/'evaluation/baseline_reference.npz';report,code=judge.evaluate(good);assert code==0 and report['passed']
    with np.load(good,allow_pickle=False) as z:x=z['reconstruction'].copy()
    np.savez_compressed(tmp_path/'zero.npz',reconstruction=np.zeros_like(x))
    report,code=judge.evaluate(tmp_path/'zero.npz');assert code==1 and report['status']=='fail'
    x[:]=.5;np.savez_compressed(tmp_path/'poor.npz',reconstruction=x)
    report,code=judge.evaluate(tmp_path/'poor.npz');assert code==1 and report['metrics']['known_pixel_max_error']==.5


@pytest.mark.parametrize('problem',['missing','key','shape','dtype','complex','nan','inf','range'])
def test_bad_submission(tmp_path,problem):
    p=tmp_path/'prediction.npz';x=np.zeros(judge.SHAPE)
    if problem=='key':np.savez(p,wrong=x)
    elif problem!='missing':
        if problem=='shape':x=x[0]
        if problem=='dtype':x=x.astype('float32')
        if problem=='complex':x=x.astype('complex128')
        if problem=='nan':x.flat[0]=np.nan
        if problem=='inf':x.flat[0]=np.inf
        if problem=='range':x.flat[0]=1.1
        np.savez(p,reconstruction=x)
    report,code=judge.evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)


def test_missing_evaluator_asset_and_configuration(monkeypatch,tmp_path):
    monkeypatch.setattr(judge,'ASSETS',tmp_path)
    report,code=judge.evaluate('missing');assert code==2 and report['metrics']=={}
    (tmp_path/'metrics.json').write_text(json.dumps(dict(ncc_boundary=float('nan'))),encoding='utf8')
    report,code=judge.evaluate('missing');assert code==2 and report['metrics']=={}


def test_cli_cwd_and_report_failure(tmp_path):
    good=ROOT/'evaluation/baseline_reference.npz';p=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(good),'--report',str(p)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True);assert result.returncode==0 and json.loads(p.read_text())['passed']
    result=subprocess.run(command[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True);assert result.returncode==2 and b'report write failed' in result.stderr


def test_named_boundaries_are_consumed(tmp_path,monkeypatch):
    source=ROOT/'evaluation';assets=tmp_path/'evaluation';assets.mkdir()
    (tmp_path/'data').mkdir();(tmp_path/'data/raw_data.npz').write_bytes((ROOT/'data/raw_data.npz').read_bytes())
    (assets/'baseline_reference.npz').write_bytes((source/'baseline_reference.npz').read_bytes())
    c=json.loads((source/'metrics.json').read_text());c['ncc_boundary']=1.01
    (assets/'metrics.json').write_text(json.dumps(c));monkeypatch.setattr(judge,'ASSETS',assets)
    report,code=judge.evaluate(source/'baseline_reference.npz');assert code==2 and 'boundary' in report['errors'][0]
    c['ncc_boundary']=1.;c['nrmse_boundary']=0;c['known_pixel_max_error_boundary']=0
    (assets/'metrics.json').write_text(json.dumps(c));report,code=judge.evaluate(source/'baseline_reference.npz');assert code==0
