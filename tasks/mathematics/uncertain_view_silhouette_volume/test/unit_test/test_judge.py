from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np,pytest
T=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',T/'evaluation/judge.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
def truth():
    with np.load(T/'evaluation/ground_truth.npz') as z:return {k:z[k].copy() for k in z.files}
def evaluate(tmp_path,values):
    p=tmp_path/'prediction.npz';np.savez_compressed(p,**values);return J.evaluate(p)
def test_true_float64_pass_and_periodic_rotation_equivalence(tmp_path):
    x={k:v.astype(np.float64) for k,v in truth().items()};x['viewpoints']+=2*np.pi;r,c=evaluate(tmp_path,x);assert c==0 and r['passed']
def test_independent_metric_definitions():
    y=truth();x={k:v.copy() for k,v in y.items()};x['reconstruction']*=.5;m=J.metrics(x,y)
    assert abs(m['density_ncc']-1)<1e-12
    expected=np.sqrt(np.mean((x['reconstruction'].astype(float)-y['reconstruction'])**2))/np.ptp(y['reconstruction'])
    assert abs(m['density_nrmse']-expected)<1e-12
def test_right_angle_geodesic_and_review_metric_agreement():
    from src.visualization import density_metrics,pose_errors
    y=truth();x={k:v.copy().astype(float) for k,v in y.items()};x['viewpoints'][:]=0;y['viewpoints']=np.zeros_like(x['viewpoints']);x['viewpoints'][0,0,0]=np.pi/2
    m=J.metrics(x,y);assert abs(m['view_max_rad']-np.pi/2)<1e-15;assert abs(m['view_rmse_rad']-np.pi/2/np.sqrt(8))<1e-15
    np.testing.assert_allclose(pose_errors(x['viewpoints'][0],y['viewpoints'][0]),[np.pi/2]+[0]*7,atol=1e-15)
    # Two float64 reduction orders over2Mvoxels differ by~2e-13 at unity.
    review=density_metrics(x['reconstruction'],y['reconstruction']);assert abs(review['ncc']-m['density_ncc'])<1e-12;assert abs(review['nrmse']-m['density_nrmse'])<1e-14
def test_cosine_scale_invariance_at_extreme_small_finite_amplitude():
    y=truth();x={k:v.astype(float) for k,v in y.items()};x['reconstruction']*=1e-308
    assert abs(J.metrics(x,y)['density_ncc']-1)<1e-12
@pytest.mark.parametrize('bad',['zero','ones','flipped','wrong_pose','prior_only'])
def test_poor_independent_quantities_fail(tmp_path,bad):
    x=truth()
    if bad=='zero':x['reconstruction']*=0
    elif bad=='ones':x['reconstruction'][:]=1
    elif bad=='flipped':x['reconstruction']=x['reconstruction'][:,:,::-1,:].copy()
    elif bad=='wrong_pose':x['viewpoints']+=.7
    else:
        with np.load(T/'data/raw_data.npz') as z:x['viewpoints']=z['view_prior']
    r,c=evaluate(tmp_path,x);assert c==1 and r['status']=='fail'
@pytest.mark.parametrize('bad',['missing','shape','integer','complex','object','nan','inf','huge','extra'])
def test_malformed(tmp_path,bad):
    x=truth()
    if bad=='missing':del x['viewpoints']
    elif bad=='shape':x['viewpoints']=x['viewpoints'][0]
    elif bad=='integer':x['viewpoints']=x['viewpoints'].astype(int)
    elif bad=='complex':x['viewpoints']=x['viewpoints'].astype(complex)
    elif bad=='object':x['viewpoints']=x['viewpoints'].astype(object)
    elif bad=='extra':x['extra']=np.array([0])
    else:x['reconstruction']=x['reconstruction'].astype(float);x['reconstruction'].flat[0]={'nan':np.nan,'inf':np.inf,'huge':1e308}[bad]
    r,c=evaluate(tmp_path,x);assert c==2 and r['status']=='error' and r['metrics']=={} and r['errors']
def test_missing_corrupt_and_evaluator_error(tmp_path,monkeypatch):
    r,c=J.evaluate(tmp_path/'missing');assert c==2 and not r['metrics']
    p=tmp_path/'bad.npz';p.write_bytes(b'bad');r,c=J.evaluate(p);assert c==2
    monkeypatch.setattr(J,'ROOT',tmp_path);r,c=J.evaluate(T/'evaluation/ground_truth.npz');assert c==2 and r['metrics']=={}
def test_cli_foreign_directory_and_report_failure(tmp_path):
    command=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'evaluation/ground_truth.npz'),'--report',str(tmp_path/'new/report.json')]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0;assert json.loads((tmp_path/'new/report.json').read_text())['passed']
    bad=tmp_path/'file';bad.write_text('x');command[-1]=str(bad/'report.json');p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==2 and 'Report write failure' in p.stderr
