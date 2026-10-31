from pathlib import Path
import importlib.util,json,subprocess,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def native(tmp_path,prediction):
    report=tmp_path/'nested/report.json'
    result=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    return result.returncode,json.loads(report.read_text(encoding='utf8'))

def valid_positions():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:return f['final'][:,:3][None]

def test_independently_known_metrics():
    assert judge.cosine(np.array([1.,0]),np.array([0.,2]))==0
    assert judge.range_nrmse(np.array([2.,2.]),np.array([0.,2.]))==pytest.approx(1/np.sqrt(2))
    assert judge.cosine(np.zeros(2),np.ones(2))==0
    with pytest.raises(ValueError):judge.range_nrmse(np.zeros(2),np.ones(2))

def test_native_foreign_working_directory_pass_and_quality_failure(tmp_path):
    p=tmp_path/'valid.npz';x=valid_positions();np.savez(p,reconstruction=x)
    code,report=native(tmp_path,p);assert code==0 and report['status']=='pass' and report['passed']
    x=x.copy();x[0,:10,2]+=2;np.savez(p,reconstruction=x)
    code,report=native(tmp_path,p);assert code==1 and report['status']=='fail' and not report['passed']

@pytest.mark.parametrize('case',['missing','keys','shape','dtype','complex','nan','inf','not_npz','gauge'])
def test_malformed_contracts(tmp_path,case):
    p=tmp_path/'bad.npz';x=valid_positions()
    if case=='missing':pass
    elif case=='keys':np.savez(p,other=x)
    elif case=='shape':np.savez(p,reconstruction=x[0])
    elif case=='dtype':np.savez(p,reconstruction=x.astype('float32'))
    elif case=='complex':np.savez(p,reconstruction=x.astype('complex128'))
    elif case=='not_npz':p.write_text('not an npz',encoding='utf8')
    else:
        x=x.copy()
        if case=='gauge':x[0,10,0]=1
        else:x[0,0,0]=np.nan if case=='nan' else np.inf
        np.savez(p,reconstruction=x)
    code,report=native(tmp_path,p);assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']

def test_evaluator_asset_failures_and_truth_precedence(tmp_path):
    e=tmp_path/'evaluation';e.mkdir();d=tmp_path/'data';d.mkdir()
    for name in ['metrics.json','ground_truth.npz']:shutil.copyfile(ROOT/'evaluation'/name,e/name)
    shutil.copyfile(ROOT/'data/raw_data.npz',d/'raw_data.npz')
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=valid_positions())
    # An unrelated method reference must never override available surveyed truth.
    np.savez(e/'baseline_reference.npz',reconstruction=np.ones((1,24,3)))
    assert judge.evaluate(p,e)['status']=='pass'
    np.savez(e/'ground_truth.npz',reconstruction=np.zeros((1,24,3)))
    r=judge.evaluate(p,e);assert r['status']=='error' and r['metrics']=={}
    (e/'ground_truth.npz').unlink();r=judge.evaluate(p,e);assert r['status']=='error' and r['metrics']=={}

def test_report_write_failure(tmp_path):
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=valid_positions())
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path)],capture_output=True)
    assert r.returncode==2 and b'Report write failed' in r.stderr
