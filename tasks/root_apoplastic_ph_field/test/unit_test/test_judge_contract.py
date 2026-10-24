from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ph_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


@pytest.fixture
def small_assets(tmp_path,monkeypatch):
    assets=tmp_path/'evaluation';assets.mkdir();data=tmp_path/'data';data.mkdir()
    reference=np.array([[[[1.,2.],[4.,0.]],[[2.,3.],[5.,0.]]]],dtype=np.float32)
    ratio=np.array([[[[.5,1.],[2.,np.nan]],[[.7,1.2],[2.1,0.]]]],dtype=np.float32)
    np.savez(assets/'baseline_reference.npz',reconstruction=reference);np.savez(data/'raw_data.npz',ratio=ratio)
    (assets/'metrics.json').write_text(json.dumps(dict(ncc_boundary=.999999,nrmse_boundary=1e-6)))
    monkeypatch.setattr(judge,'HERE',assets);monkeypatch.setattr(judge,'SHAPE',reference.shape)
    return reference,tmp_path/'prediction.npz'


@pytest.mark.parametrize('kind',['missing','zip','key','extra','shape','int','complex','float16','object','nan','inf','outside'])
def test_invalid_prediction_errors_keep_metrics_empty(kind,small_assets):
    x,path=small_assets;x=x.copy()
    if kind=='missing':pass
    elif kind=='zip':path.write_bytes(b'broken')
    elif kind=='key':np.savez(path,wrong=x)
    elif kind=='extra':np.savez(path,reconstruction=x,extra=x)
    else:
        if kind=='shape':x=x[0]
        if kind=='int':x=x.astype(int)
        if kind=='complex':x=x.astype(complex)
        if kind=='float16':x=x.astype(np.float16)
        if kind=='object':x=np.array([object()],dtype=object)
        if kind=='nan':x.flat[0]=np.nan
        if kind=='inf':x.flat[0]=np.inf
        if kind=='outside':x[0,0,1,1]=1
        np.savez(path,reconstruction=x)
    result=judge.evaluate(path);assert result['status']=='error' and result['metrics']=={} and result['errors']


@pytest.mark.parametrize('kind',['zero','negative','constant','scale','one_bad_root'])
def test_bad_physical_results_and_single_failed_root_fail(kind,small_assets):
    x,path=small_assets;x=x.copy();mask=x!=0
    if kind=='zero':x[:]=0
    elif kind=='negative':x=-x
    elif kind=='constant':x[mask]=3
    elif kind=='scale':x*=1.1
    else:x[0,1]*=1.1
    np.savez(path,reconstruction=x);result=judge.evaluate(path)
    assert result['status']=='fail' and not result['passed'] and result['metrics']


def test_known_metric_case_and_precision(small_assets):
    x,path=small_assets;x=x.astype(float);np.savez(path,reconstruction=x)
    assert judge.evaluate(path)['passed']
    x[0,0,1,0]=3;np.savez(path,reconstruction=x);result=judge.evaluate(path)
    assert result['metrics']['image_00_ncc']==pytest.approx(17/np.sqrt(14*21))
    assert result['metrics']['image_00_nrmse']==pytest.approx(np.sqrt(1/3)/3)


def test_evaluator_errors(small_assets):
    x,path=small_assets;np.savez(path,reconstruction=x)
    (judge.HERE/'baseline_reference.npz').unlink()
    result=judge.evaluate(path);assert result['status']=='error' and result['errors'][0].startswith('Evaluator error:')
    (judge.HERE/'metrics.json').write_text(json.dumps(dict(ncc_boundary=None,nrmse_boundary=1e-6)))
    assert judge.evaluate(path)['metrics']=={}


def test_full_native_foreign_cwd_cli_exit_codes_and_report_failure(tmp_path):
    # Full native assets establish path semantics; small fixtures test every malformed branch.
    prediction=ROOT/'output/reconstruction.npz';report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert run.returncode==0
    bad=tmp_path/'bad.npz';np.savez_compressed(bad,reconstruction=np.zeros(judge.SHAPE,dtype=np.float32))
    cmd[cmd.index('--prediction')+1]=str(bad)
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==1
    cmd[cmd.index('--prediction')+1]=str(tmp_path/'missing.npz')
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==2
    block=tmp_path/'block';block.write_text('file');cmd[-1]=str(block/'child.json')
    run=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert run.returncode==2 and b'Could not write report' in run.stderr
