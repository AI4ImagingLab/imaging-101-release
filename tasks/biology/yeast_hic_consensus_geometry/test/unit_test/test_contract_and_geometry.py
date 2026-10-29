"""Independent geometry, public-input and hostile-submission checks."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

@pytest.fixture
def reference():
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:return f['reconstruction'].copy()

def save(tmp_path,x,**kw):
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=x,**kw);return p

def test_known_distance_metrics():
    r=np.array([[0.,0,0],[1,0,0],[0,2,0]])
    m=np.ones(3,dtype=bool);v=judge.distance_metrics(2*r,r,m)
    assert v['distance_ncc']==pytest.approx(1)
    d=np.array([1,2,np.sqrt(5)])
    assert v['distance_nrmse']==pytest.approx(np.sqrt(np.mean(d*d))/np.ptp(d))

def test_global_rigid_reflection_invariance(tmp_path,reference):
    with np.load(ROOT/'data/raw_data.npz') as f:m=f['bead_support'][0]
    x=reference.copy();x[0,m]=x[0,m]@np.array([[0.,1,0],[1,0,0],[0,0,1]])+np.array([9.,-3,7])
    report,code=judge.evaluate(save(tmp_path,x));assert code==0,report

@pytest.mark.parametrize('kind',['scale','collapse','permutation','chromosome_translation'])
def test_bad_geometry_fails(tmp_path,reference,kind):
    with np.load(ROOT/'data/raw_data.npz') as f:m=f['bead_support'][0];chrom=f['chromosome_index'][0]
    x=reference.copy()
    if kind=='scale':x*=2
    elif kind=='collapse':x*=0
    elif kind=='permutation':x[0,m]=x[0,m][::-1]
    else:x[0,m&(chrom==0)]+=np.array([2,0,0])
    report,code=judge.evaluate(save(tmp_path,x));assert code==1 and report['metrics'],report

@pytest.mark.parametrize('kind',['shape','int','complex','nan','inf','padding','extra','missing','object','overflow'])
def test_invalid_submission_errors(tmp_path,reference,kind):
    x=reference.copy();extra={}
    if kind=='shape':x=x[0]
    elif kind=='int':x=x.astype(np.int64)
    elif kind=='complex':x=x.astype(complex)
    elif kind=='nan':x[0,0,0]=np.nan
    elif kind=='inf':x[0,0,0]=np.inf
    elif kind=='padding':x[0,21,0]=1
    elif kind=='extra':extra={'other':np.zeros(1)}
    elif kind=='object':x=x.astype(object)
    elif kind=='overflow':x*=1e300
    p=save(tmp_path,x,**extra)
    if kind=='missing':np.savez(p,other=x)
    report,code=judge.evaluate(p)
    assert code==2 and report['status']=='error' and report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)

def test_missing_prediction(tmp_path):
    r,c=judge.evaluate(tmp_path/'absent.npz');assert c==2 and r['metrics']=={}

def test_missing_evaluator_asset(tmp_path,reference,monkeypatch):
    monkeypatch.setattr(judge,'REFERENCE',tmp_path/'missing.npz')
    r,c=judge.evaluate(save(tmp_path,reference));assert c==2 and r['metrics']=={}

def test_invalid_metric_configuration(tmp_path,reference,monkeypatch):
    p=tmp_path/'metrics.json';d=json.loads(judge.METRICS.read_text());d['relative_poisson_excess_boundary']=float('nan');p.write_text(json.dumps(d));monkeypatch.setattr(judge,'METRICS',p)
    r,c=judge.evaluate(save(tmp_path,reference));assert c==2 and r['metrics']=={}

def test_foreign_cwd_and_report_failure(tmp_path,reference):
    prediction=save(tmp_path,reference);report=tmp_path/'nested'/'report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0 and json.loads(report.read_text())['passed']
    p=subprocess.run(command[:-1]+[str(prediction/'impossible.json')],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==2 and 'Cannot write judge report' in p.stderr

def test_aggregation_conserves_contacts_and_boundaries():
    from src.preprocessing import aggregate_contacts
    c=np.zeros((5,5),np.int64);c[0,1]=2;c[0,2]=3;c[1,4]=7;c[3,4]=11
    out,chrom,mapping,lost=aggregate_contacts(c,np.array([0,0,0,1,1]),2)
    assert out.sum()+lost==c.sum() and lost==13
    assert np.array_equal(mapping,[0,0,1,2,2]) and np.array_equal(chrom,[0,0,1])

def test_isotonic_ties_and_order():
    from src.solvers import monotone_fit
    x=np.array([3.,1,2,2]);y=np.array([0.,2,3,1]);z=monotone_fit(x,y)
    assert z[2]==z[3] and np.all(np.diff(z[np.argsort(x)])>=0)
    assert np.allclose(z,1.5)

def test_public_schema_and_no_source_url():
    from src.preprocessing import load_data
    data=load_data(ROOT/'data/raw_data.npz')
    assert sum(data['bead_support'][0])==343
    assert np.array_equal(np.bincount(data['chromosome_index'][0]),[24,82,32,154,58])
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for forbidden in ['github.com','githubusercontent.com','github.io','git.io','baseline_reference.npz']:
        assert forbidden not in readme.lower()
    for key in data:assert '| '+key+' |' in readme or '| `'+key+'` |' in readme
