from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.physics_model import gaussian,physical_columns
from src.solvers import fit_one,model_image
from src.preprocessing import PIXEL_NM,load_data
T=Path(__file__).parents[2]
s=importlib.util.spec_from_file_location('judge',T/'evaluation/judge.py');j=importlib.util.module_from_spec(s);s.loader.exec_module(j)
@pytest.fixture
def baseline():
    with np.load(T/'evaluation/baseline_reference.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
@pytest.mark.parametrize('i',[0,1,2])
def test_independent_known_rotated_gaussian(i):
    z=np.load(T/'test/fixtures/known_original_controls.npz',allow_pickle=False)
    p=z[f'case{i}_expected'];cx,cy,sx,sy,theta,A=p
    c,s=np.cos(theta),np.sin(theta);r=np.array([[c,-s],[s,c]]);cov=r.T@np.diag([sx*sx,sy*sy])@r
    yy,xx=np.mgrid[:65,:65];d=np.stack((xx-cx,yy-cy),axis=-1)
    expected=A/(2*np.pi*np.sqrt(np.linalg.det(cov)))*np.exp(-.5*np.einsum('...i,ij,...j->...',d,np.linalg.inv(cov),d))
    assert_allclose(gaussian([A,sx,sy,cx,cy,theta],xx,yy),expected,atol=1e-12,rtol=1e-12)
    fit,ok=fit_one(z[f'case{i}_image'],np.array([cx+.3,cy-.2,1.4,1.4,.01,1]),8)
    assert ok and np.linalg.norm(fit[:2]-p[:2])<.005
    actual=physical_columns(fit[None],1)[0]
    assert np.linalg.norm(actual[2:5]-[cov[0,0],cov[0,1],cov[1,1]])/np.linalg.norm(cov)<.02
def test_covariance_units_and_equivalent_axes():
    a=np.array([[12.,14.,1.2,1.8,.4,300.]])
    b=a.copy();b[:,2:4]=a[:,[3,2]];b[:,4]+=np.pi/2
    assert_allclose(physical_columns(a,PIXEL_NM),physical_columns(b,PIXEL_NM),atol=1e-15)
    c=physical_columns(a,PIXEL_NM)[0]
    assert_allclose(c[:2],a[0,:2]*PIXEL_NM)
    assert c[2]*c[4]-c[3]**2>0
def test_exact_metric_cases_and_permutation():
    b=np.array([[0,0,1,0,1,1],[1,0,1,0,1,2.]],float)
    a=b.copy();a[:,:2]+=[PIXEL_NM,0];a[:,2:5]*=1.5;a[:,5]*=2
    r=j.compare(a,b)
    assert_allclose([r['centroid_max_pixel'],r['centroid_rms_pixel'],r['covariance_relative'],r['peak_relative']],[1,1,.5,1])
    assert all(v==0 for v in j.compare(b[::-1],b).values())
@pytest.mark.parametrize('kind',['missing','key','shape','dtype','complex','nan','inf','object'])
def test_malformed_inputs(tmp_path,baseline,kind):
    path=tmp_path/'bad.npz'
    if kind=='missing':pass
    else:
        a={k:v.copy() for k,v in baseline.items()};k='oxygen_columns'
        if kind=='key':del a[k]
        elif kind=='shape':a[k]=a[k][:,:-1]
        elif kind=='dtype':a[k]=a[k].astype(np.float32)
        elif kind=='complex':a[k]=a[k].astype(complex)
        elif kind=='object':a[k]=a[k].astype(object)
        else:a[k][0,0,0]=np.nan if kind=='nan' else np.inf
        np.savez(path,**a)
    r,code=j.evaluate(path)
    assert code==2 and r['status']=='error' and not r['metrics'] and r['errors']
    json.dumps(r,allow_nan=False)
@pytest.mark.parametrize('kind',['centres','covariance','peak','invalid_covariance','negative_peak'])
def test_each_field_rejected_when_poor(tmp_path,baseline,kind):
    a={k:v.copy() for k,v in baseline.items()};k='b_columns'
    if kind=='centres':a[k][:,:,:2]+=.5*PIXEL_NM
    elif kind=='covariance':a[k][:,:,2:5]*=1.01
    elif kind=='peak':a[k][:,:,5]*=1.01
    elif kind=='invalid_covariance':a[k][0,0,2]=-1
    else:a[k][0,0,5]=-1e-20
    path=tmp_path/'poor.npz';np.savez(path,**a);r,code=j.evaluate(path)
    assert code==1 and r['status']=='fail' and not r['passed']
def test_valid_reference_and_unordered_columns(tmp_path,baseline):
    np.savez(tmp_path/'good.npz',**{k:v[:,::-1] for k,v in baseline.items()})
    r,code=j.evaluate(tmp_path/'good.npz');assert code==0 and r['passed']
@pytest.mark.parametrize('kind',['missing_reference','invalid_truth','negative_ref','bad_boundary','boolean_boundary','nan_boundary','invalid_physical_boundary'])
def test_evaluator_errors_and_truth_priority(tmp_path,baseline,kind):
    path=tmp_path/'good.npz';np.savez(path,**baseline)
    ref=tmp_path/'hidden';ref.mkdir()
    config=json.loads((T/'evaluation/metrics.json').read_text())
    if kind=='bad_boundary':config['peak_relative_boundary']=-1
    if kind=='boolean_boundary':config['centroid_max_pixel_boundary']=True
    if kind=='nan_boundary':config['centroid_rms_pixel_boundary']=float('nan')
    if kind=='invalid_physical_boundary':config['invalid_profile_count_boundary']=1
    (ref/'metrics.json').write_text(json.dumps(config))
    if kind!='missing_reference':np.savez(ref/'baseline_reference.npz',**baseline)
    if kind=='invalid_truth':np.savez(ref/'ground_truth.npz',wrong=np.ones(3))
    if kind=='negative_ref':
        bad={k:v.copy() for k,v in baseline.items()};bad['a_columns'][0,0,5]=-1;np.savez(ref/'ground_truth.npz',**bad)
    r,code=j.evaluate(path,ref);assert code==2 and not r['metrics'] and r['errors']
def test_foreign_cwd_cli_and_report_write_failure(tmp_path):
    good=T/'evaluation/baseline_reference.npz';report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(good),'--report',str(report)]
    r=subprocess.run(cmd,cwd=tmp_path,capture_output=True)
    assert r.returncode==0 and json.loads(report.read_text())['passed']
    r=subprocess.run(cmd[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True)
    assert r.returncode==2 and b'Cannot write report' in r.stderr
def test_finite_extreme_input_never_emits_nonfinite_json(tmp_path,baseline):
    a={k:v.copy() for k,v in baseline.items()};a['a_columns'][0,0,2:5]=np.finfo(float).max
    np.savez(tmp_path/'huge.npz',**a);r,code=j.evaluate(tmp_path/'huge.npz')
    assert code==2 and not r['metrics'];json.dumps(r,allow_nan=False)
def test_data_and_readme_calibration():
    data=load_data(T/'data/raw_data.npz');assert set(data)=={'adf','abf'}
    assert str(PIXEL_NM) in (T/'README.md').read_text(encoding='utf8')
