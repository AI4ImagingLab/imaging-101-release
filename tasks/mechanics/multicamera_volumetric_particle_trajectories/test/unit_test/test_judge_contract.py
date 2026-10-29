from pathlib import Path
import json,sys,subprocess,numpy as np,pytest
def sample():
    f=np.tile(np.arange(100,dtype=np.int16),2);ids=np.repeat(np.arange(2,dtype=np.int32),100);p=np.c_[f*.001,ids.astype(float),np.zeros(200)];return p,f,ids
def save(path,p=None,f=None,t=None):
    a,b,c=sample();np.savez(path,positions=(a if p is None else p)[None],frame_index=(b if f is None else f)[None],track_id=(c if t is None else t)[None]);return path
def test_analytic_metrics_id_permutation_row_order(judge):
    x=sample();expected={'point_recall':1.,'point_precision':1.,'position_rms_mm':0.,'edge_recall':1.,'edge_precision':1.};assert judge.measure(x,x)==expected
    p,f,t=x;order=np.arange(len(p))[::-1];assert judge.measure((p[order],f[order],(7-t)[order]),x)==expected
def test_singletons_and_wrong_connections_fail(judge):
    p,f,t=sample();m=judge.measure((p,f,np.arange(len(p),dtype=np.int32)),(p,f,t));assert m['point_recall']==1 and m['edge_recall']==m['edge_precision']==0
    altered=t.copy();altered[50:150]=1-altered[50:150];m=judge.measure((p,f,altered),(p,f,t));assert m['edge_precision']<1
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','frame','negative_id','duplicate'])
def test_invalid_submission_is_finite_error(tmp_path,judge,kind):
    path=tmp_path/'bad.npz';p,f,t=sample()
    if kind=='missing':pass
    elif kind=='key':np.savez(path,positions=p[None])
    else:
        if kind=='shape':p=p[:-1]
        elif kind=='dtype':p=p.astype(np.float32)
        elif kind=='complex':p=p.astype(np.complex128)
        elif kind=='nan':p[0,0]=np.nan
        elif kind=='inf':p[0,0]=np.inf
        elif kind=='frame':f[0]=100
        elif kind=='negative_id':t[0]=-1
        elif kind=='duplicate':f[1]=f[0]
        save(path,p,f,t)
    report=judge.evaluate(path);assert report['status']=='error' and report['metrics']=={} and report['errors'];json.dumps(report,allow_nan=False)
def test_extreme_finite_estimate_is_writable_physical_failure(tmp_path,judge):
    p,f,t=sample();p[:]=1e300;path=save(tmp_path/'large.npz',p,f,t);report=judge.evaluate(path);assert report['status']=='fail' and not report['passed'];assert all(np.isfinite(v)for v in report['metrics'].values());json.dumps(report,allow_nan=False)
@pytest.mark.parametrize('key,value',[('point_recall_boundary',0),('point_precision_boundary',1.1),('edge_recall_boundary',-1),('edge_precision_boundary',True),('position_rms_mm_boundary',-.1),('position_rms_mm_boundary',float('nan')),('matching_radius_mm',.2)])
def test_invalid_metric_configuration(tmp_path,judge,key,value):
    x=json.loads((judge.HERE/'metrics.json').read_text());x[key]=value;p=tmp_path/'metrics.json';p.write_text(json.dumps(x))
    with pytest.raises(ValueError):judge.read_limits(p)
def test_actual_truth_priority_invalid_truth_no_fallback(tmp_path,judge):
    (tmp_path/'metrics.json').write_bytes((judge.HERE/'metrics.json').read_bytes());save(tmp_path/'baseline_reference.npz');save(tmp_path/'prediction.npz');np.savez(tmp_path/'ground_truth.npz',invalid=np.zeros(1));report=judge.evaluate(tmp_path/'prediction.npz',tmp_path);assert report['status']=='error' and 'Missing required' in report['errors'][0]
def test_report_write_failure_and_foreign_cwd(tmp_path,judge):
    prediction=tmp_path/'missing.npz';report=tmp_path/'nested/report.json';r=subprocess.run([sys.executable,str(judge.HERE/'judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True);assert r.returncode==2 and json.loads(report.read_text())['status']=='error'
    occupied=tmp_path/'file';occupied.write_text('x');r=subprocess.run([sys.executable,str(judge.HERE/'judge.py'),'--prediction',str(prediction),'--report',str(occupied/'r.json')],cwd=tmp_path,capture_output=True);assert r.returncode==2 and b'Report-write failure' in r.stderr
