from pathlib import Path
import importlib.util,json,os,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def case(tmp_path_factory):
    folder=tmp_path_factory.mktemp('judge');ev=folder/'evaluation';ev.mkdir();(folder/'data').mkdir()
    os.link(ROOT/'data/raw_data.npz',folder/'data/raw_data.npz')
    for name in ['metrics.json','baseline_reference.npz']:shutil.copy2(ROOT/'evaluation'/name,ev/name)
    spec=importlib.util.spec_from_file_location('independent_mwf_judge',ROOT/'evaluation/judge.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=ev
    with np.load(ROOT/'output/reconstruction.npz') as z:good={k:z[k] for k in z.files}
    return m,folder,good

def call(case,pred):
    m,f,_=case;np.savez(f/'prediction.npz',**pred);code=m.main(['--prediction',str(f/'prediction.npz'),'--report',str(f/'report.json')]);return code,json.loads((f/'report.json').read_text())

def test_good_and_foreign_cwd(case,monkeypatch):
    monkeypatch.chdir(case[1]);code,r=call(case,case[2]);assert code==0 and r['passed'] and len(r['metrics'])==6

@pytest.mark.parametrize('key',['objective_sum_boundary','scaled_duality_gap_boundary','derived_consistency_boundary'])
@pytest.mark.parametrize('bad',[True,False,'0.1',[],{},None,float('nan'),float('inf'),-float('inf'),-1.,0.])
def test_bad_metric_config(case,key,bad):
    m,f,p=case;path=m.HERE/'metrics.json';saved=path.read_text();c=json.loads(saved);c[key]=bad;path.write_text(json.dumps(c))
    try:
        code,r=call(case,p);assert code==2 and r['status']=='error' and r['metrics']=={} and r['errors']
    finally:path.write_text(saved)

@pytest.mark.parametrize('key',['objective_sum_boundary','scaled_duality_gap_boundary','derived_consistency_boundary'])
def test_missing_metric_config_key(case,key):
    m,f,p=case;path=m.HERE/'metrics.json';saved=path.read_text();c=json.loads(saved);del c[key];path.write_text(json.dumps(c))
    try:assert call(case,p)[0]==2
    finally:path.write_text(saved)

@pytest.mark.parametrize('kind',['missing','shape','float32','complex','integer','nan','inf','negative','object','mass_bound'])
def test_malformed_arrays(case,kind):
    p={k:v.copy() for k,v in case[2].items()};key='reconstruction'
    if kind=='missing':del p[key]
    elif kind=='shape':p[key]=p[key][0]
    elif kind=='float32':p[key]=p[key].astype(np.float32)
    elif kind=='complex':p[key]=p[key].astype(complex)
    elif kind=='integer':p[key]=p[key].astype(int)
    elif kind=='object':p[key]=np.array([object()],object)
    else:p[key].flat[0]={'nan':np.nan,'inf':np.inf,'negative':-1.,'mass_bound':1e300}[kind]
    code,r=call(case,p);assert code==2 and r['metrics']=={} and r['errors']

@pytest.mark.parametrize('key',['reconstruction','myelin_fraction','total_density','predicted_echoes'])
def test_each_output_is_scored(case,key):
    p={k:v.copy() for k,v in case[2].items()}
    if key=='reconstruction':p[key]*=.5
    else:p[key]*=.9
    code,r=call(case,p);assert code==1 and not r['passed'] and r['metrics']

def test_zero_solution_fails_quality_and_certificate(case):
    code,r=call(case,{k:np.zeros_like(v) for k,v in case[2].items()});assert code==1 and r['metrics']['objective_sum']>r['thresholds']['objective_sum']['boundary']

def test_actual_truth_precedes_bad_baseline(case):
    m,f,p=case;baseline=m.HERE/'baseline_reference.npz';saved=baseline.read_bytes();shutil.copy2(baseline,m.HERE/'ground_truth.npz');np.savez(baseline,wrong=np.zeros(1))
    try:assert call(case,p)[0]==0
    finally:(m.HERE/'ground_truth.npz').unlink();baseline.write_bytes(saved)

def test_missing_evaluator_assets_and_prediction(case):
    m,f,p=case;baseline=m.HERE/'baseline_reference.npz';saved=baseline.read_bytes();baseline.unlink()
    try:
        code,r=call(case,p);assert code==2 and r['metrics']=={}
    finally:baseline.write_bytes(saved)
    code=m.main(['--prediction',str(f/'absent.npz'),'--report',str(f/'missing.json')]);assert code==2

def test_report_write_failure(case):
    m,f,p=case;np.savez(f/'prediction.npz',**p);blocked=f/'blocked';blocked.write_text('file')
    assert m.main(['--prediction',str(f/'prediction.npz'),'--report',str(blocked/'report.json')])==2

def test_judge_dictionary_matches_independent_source(case):
    m,_,_=case
    with np.load(ROOT/'data/raw_data.npz') as z:centers=z['cluster_centroids'][0]
    A,t2=m.independent_dictionary(centers)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:expected=z['capture_dictionary']
    np.testing.assert_allclose(A,np.transpose(expected,(2,0,1)),atol=2e-12,rtol=2e-11)

@pytest.mark.parametrize('x',[np.array([4/3,0.]),np.array([0.,0.]),np.array([.5,2.])])
def test_judge_analytic_duality_and_all_derived_metrics(case,x):
    m,_,_=case;A=np.diag([2.,1.]);y=np.array([3.,-1.]);lam=m.LAMBDA
    # Use this task's actual lambda with a separately solved two-bin case.
    u=A@x-y;g=A.T@u;J=u@u+lam*(x@x);D=-u@u-2*y@u-np.minimum(g,0)@np.minimum(g,0)/lam
    data={'estimation_mask':np.array([[[True,False]]]),'cluster_labels':np.array([[[1,0]]]),'echo_magnitudes':np.array([[[[3.],[-1.]],[[0.],[0.]]]])[None]}
    total=x.sum();fraction=x[0]/total if total else 0
    out={'reconstruction':np.array([[x,[0.,0.]]]),'total_density':np.array([[total,0.]]),'myelin_fraction':np.array([[fraction,0.]]),'predicted_echoes':np.array([[A@x,[0.,0.]]])}
    r=m.measure(out,data,A[None],np.array([20.,80.]))
    np.testing.assert_allclose(r['objective_sum'],J,atol=1e-12)
    np.testing.assert_allclose(r['scaled_duality_gap'],(J-D)/max(J,1e-6),rtol=1e-12,atol=1e-12)
    assert r['total_density_consistency']==r['myelin_fraction_consistency']==r['predicted_echoes_consistency']==r['outside_mask_max']==0
