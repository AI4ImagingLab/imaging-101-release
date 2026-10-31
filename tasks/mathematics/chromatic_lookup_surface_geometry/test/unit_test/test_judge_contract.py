from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

def judge():
    s=importlib.util.spec_from_file_location('independent_depth_judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_independently_known_metric_definitions():
    m=judge();s=m.pair_metrics([3,2,1],[1,2,3]);assert abs(s['ncc']-10/14)<1e-15
    assert abs(s['nrmse']-np.sqrt(8/3)/2)<1e-15 and s['maximum_absolute_error_mm']==2
    with pytest.raises(ValueError):m.pair_metrics([0,0],[0,0])

@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','infinity','pickle','sentinel'])
def test_malformed_submissions(tmp_path,kind):
    m=judge();p=tmp_path/'prediction.npz'
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=np.zeros(m.SHAPE,dtype=np.float32))
    elif kind=='shape':np.savez(p,reconstruction=np.zeros((3,650,650),dtype=np.float32))
    elif kind=='dtype':np.savez(p,reconstruction=np.zeros(m.SHAPE,dtype=np.float64))
    elif kind=='complex':np.savez(p,reconstruction=np.zeros(m.SHAPE,dtype=np.complex64))
    elif kind=='pickle':np.savez(p,reconstruction=np.array([object()]))
    else:
        with np.load(m.REFERENCE) as f:a=f['reconstruction']
        a.flat[0]=np.nan if kind=='nan' else np.inf if kind=='infinity' else 10
        np.savez(p,reconstruction=a)
    report,code=m.evaluate(p);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_valid_but_poor_field_fails(tmp_path):
    m=judge()
    with np.load(m.REFERENCE) as f:a=f['reconstruction']
    a[a!=-1]=900;path=tmp_path/'poor.npz';np.savez(path,reconstruction=a)
    report,code=m.evaluate(path);assert code==1 and report['status']=='fail' and not report['passed'] and len(report['metrics'])==12

def test_actual_truth_priority_and_no_invalid_truth_fallback(tmp_path,monkeypatch):
    m=judge()
    with np.load(m.REFERENCE) as f:a=f['reconstruction']
    a[a!=-1]+=.01;path=tmp_path/'prediction.npz';truth=tmp_path/'truth.npz';np.savez(path,reconstruction=a);np.savez(truth,reconstruction=a)
    monkeypatch.setattr(m,'TRUTH',truth)
    report,code=m.evaluate(path);assert code==0 and all(report['metrics'][f'frame{i}_nrmse']==0 for i in range(3))
    np.savez(truth,wrong=a);report,code=m.evaluate(path);assert code==2 and report['metrics']=={} and report['errors']

def test_missing_hidden_assets_are_evaluator_error(tmp_path,monkeypatch):
    m=judge();monkeypatch.setattr(m,'REFERENCE',tmp_path/'missing.npz');monkeypatch.setattr(m,'TRUTH',tmp_path/'absent.npz')
    report,code=m.evaluate(ROOT/'output/reconstruction.npz');assert code==2 and report['errors'][0].startswith('Evaluator error:')

@pytest.mark.parametrize('field,value',[('ncc_boundary',float('nan')),('nrmse_boundary',-1),('ncc_boundary',True),('observation_rms_boundaries',[1,1])])
def test_invalid_metric_configuration(tmp_path,monkeypatch,field,value):
    m=judge();config=json.loads(m.METRICS.read_text());config[field]=value;p=tmp_path/'metrics.json';p.write_text(json.dumps(config));monkeypatch.setattr(m,'METRICS',p)
    report,code=m.evaluate(ROOT/'output/reconstruction.npz');assert code==2 and report['metrics']=={}

def test_cli_from_foreign_directory_and_report_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['status']=='pass'
    (tmp_path/'block').write_text('regular file');command[-1]=str(tmp_path/'block/report.json')
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert result.returncode==2 and 'Report write failed' in result.stderr

def test_importing_main_preserves_notebook_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
