from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)


def test_independent_relative_error_and_overflow():
    x=np.array([3.,4.]);assert judge.relative_error(1.1*x,x)==pytest.approx(.1)
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError):judge.relative_error(np.full(2,1e300),x)
    with pytest.raises(ValueError):judge.relative_error(x,np.zeros(2))


def test_missing_malformed_and_evaluator_assets(tmp_path):
    for arrays in [{},{'reconstruction':np.zeros((1,))},{'reconstruction':np.zeros((1,84,240,320),dtype=np.float32),'sparse_component':np.zeros((1,))}]:
        p=tmp_path/'bad.npz';np.savez_compressed(p,**arrays);r,code=judge.evaluate(p);assert code==2 and r['metrics']=={} and r['errors']
    assert judge.evaluate(tmp_path/'absent')[1]==2
    assert judge.evaluate(tmp_path/'absent',tmp_path/'no-evaluator')[1]==2


@pytest.mark.parametrize('key',['reconstruction','sparse_component'])
def test_each_required_quantity_and_nonfinite(tmp_path,key):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a[key]=np.zeros_like(a[key]);p=tmp_path/'poor.npz';np.savez_compressed(p,**a)
    r,code=judge.evaluate(p);assert code==1 and r['status']=='fail' and r['metrics'][key+'_relative_l2']==pytest.approx(1.)
    a[key].flat[0]=np.nan;np.savez_compressed(p,**a);r,code=judge.evaluate(p);assert code==2 and not r['metrics']
    a[key].flat[0]=np.inf;np.savez_compressed(p,**a);assert judge.evaluate(p)[1]==2


def test_foreign_cwd_cli_report_and_write_error(tmp_path):
    prediction=ROOT/'evaluation/baseline_reference.npz';report=tmp_path/'nested/result.json';cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==0,p.stderr
    r=json.loads(report.read_text());assert r['passed'] and all(np.isfinite(v) for v in r['metrics'].values())
    cmd[-1]=str(tmp_path);p=subprocess.run(cmd,cwd=tmp_path,capture_output=True);assert p.returncode==2 and b'Cannot write report' in p.stderr


def test_no_backend_mutation_and_readme_constants():
    import matplotlib
    backend=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==backend
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for s in ['(1, 84, 240, 320)','uint8','float64','29.97002997002997','255','76800']:assert s in text


def test_actual_truth_priority_over_invalid_baseline(tmp_path):
    import os
    assets=tmp_path/'evaluation';assets.mkdir();data=tmp_path/'data';data.mkdir()
    os.link(ROOT/'evaluation/baseline_reference.npz',assets/'ground_truth.npz')
    os.link(ROOT/'data/raw_data.npz',data/'raw_data.npz')
    (assets/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'),encoding='utf8')
    (assets/'baseline_reference.npz').write_bytes(b'Invalid lower priority reference')
    assert judge.evaluate(ROOT/'evaluation/baseline_reference.npz',assets)[1]==0


def test_invalid_threshold_configuration(tmp_path):
    c=json.loads((ROOT/'evaluation/metrics.json').read_text(encoding='utf8'))
    for boundary in [0.,-1.,float('nan')]:
        c['additive_consistency_relative_boundary']=boundary
        (tmp_path/'metrics.json').write_text(json.dumps(c),encoding='utf8')
        r,code=judge.evaluate(ROOT/'evaluation/baseline_reference.npz',tmp_path);assert code==2 and r['metrics']=={}
