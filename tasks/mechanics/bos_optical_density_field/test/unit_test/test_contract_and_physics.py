from pathlib import Path
import importlib.util,json,shutil,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import load_observations
from src.generate_data import add_displacement_noise,conditional_density_samples
from src.physics_model import calibrate,integration_operators,REFRACTIVE_AMBIENT,PIXEL_PITCH,GLADSTONE_DALE,DEPTH,MAGNIFICATION,DISTANCE
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
def test_forward_adjoint():
 x,y=np.meshgrid(np.arange(5)*.001,np.arange(4)*.002);D,B,C=integration_operators(x,y);rng=np.random.default_rng(15);a=rng.normal(size=20);b=rng.normal(size=D.shape[0]);np.testing.assert_allclose((D@a)@b,a@(D.T@b),rtol=2e-15)
def test_linear_physics():
 f=np.load(ROOT/'test/fixtures/upstream_known_linear.npz',allow_pickle=False);D,B,C=integration_operators(f['X'],f['Y']);g=np.r_[f['gradient_x'].ravel(),f['gradient_y'].ravel()];mask=f['boundary'];rho=f['known_linear_density'];np.testing.assert_allclose(D@rho.ravel(),B@g+C@rho[mask],atol=3e-13,rtol=3e-15)
def test_optical_units_and_derivative():
 c=REFRACTIVE_AMBIENT*PIXEL_PITCH/(GLADSTONE_DALE*DEPTH*MAGNIFICATION*DISTANCE);h=1e-5
 grad=lambda u:calibrate(np.ones((2,2)),np.ones((2,2)),np.full((2,2),u),np.ones((2,2)),np.ones((2,2)),np.ones((2,2)))[2]
 np.testing.assert_allclose((grad(1+h)-grad(1-h))/(2*h),c,rtol=1e-10)
def test_noise_generator():
 a=load_observations(ROOT/'data/raw_data.npz');b=add_displacement_noise(a,seed=7);c=add_displacement_noise(a,seed=7)
 np.testing.assert_array_equal(b['displacement'],c['displacement']);np.testing.assert_array_equal(a['positions'],b['positions']);assert not np.array_equal(a['displacement'],b['displacement'])
 np.testing.assert_array_equal(add_displacement_noise(a,0)['displacement'],a['displacement'])
@pytest.mark.parametrize('value',[-1,np.nan,np.inf,True])
def test_invalid_noise(value):
 with pytest.raises(ValueError):add_displacement_noise(load_observations(ROOT/'data/raw_data.npz'),value)
def evaluator(tmp_path):
 ev=tmp_path/'evaluation';ev.mkdir();shutil.copytree(ROOT/'data',tmp_path/'data');shutil.copy2(ROOT/'evaluation/metrics.json',ev/'metrics.json');shutil.copy2(ROOT/'evaluation/baseline_reference.npz',ev/'baseline_reference.npz');return ev
@pytest.mark.parametrize('case',['missing','wrong_key','shape','integer','complex','nan','inf','negative_sigma','huge'])
def test_invalid_prediction(tmp_path,case):
 ev=evaluator(tmp_path);pred=tmp_path/'p.npz';ref=np.load(ev/'baseline_reference.npz');a={k:ref[k].copy() for k in judge.KEYS}
 if case=='missing':pass
 elif case=='wrong_key':np.savez(pred,other=a['reconstruction'])
 else:
  if case=='shape':a['reconstruction']=a['reconstruction'][0]
  if case=='integer':a['reconstruction']=a['reconstruction'].astype(int)
  if case=='complex':a['reconstruction']=a['reconstruction'].astype(complex)
  if case=='nan':a['reconstruction'][0,0,0]=np.nan
  if case=='inf':a['reconstruction'][0,0,0]=np.inf
  if case=='negative_sigma':a['density_sigma'][0,0,0]=-1
  if case=='huge':a['reconstruction'][:]=1e308
  np.savez(pred,**a)
 report=judge.evaluate(pred,ev);assert report['status']=='error' and report['metrics']=={}
@pytest.mark.parametrize('value',[True,'0.9',-1,np.inf,np.nan])
def test_invalid_threshold(tmp_path,value):
 ev=evaluator(tmp_path);config=json.loads((ev/'metrics.json').read_text());config['quantities']['reconstruction']['nrmse_boundary']=value;(ev/'metrics.json').write_text(json.dumps(config));report=judge.evaluate(ev/'baseline_reference.npz',ev);assert report['status']=='error' and report['metrics']=={}
def test_truth_precedence_and_invalid_truth(tmp_path):
 ev=evaluator(tmp_path);ref=np.load(ev/'baseline_reference.npz');truth={k:ref[k].copy() for k in judge.KEYS};truth['reconstruction'][0,16,16]+=.1;np.savez(ev/'ground_truth.npz',**truth)
 report=judge.evaluate(ev/'baseline_reference.npz',ev);assert report['status']=='fail' and report['metrics']['reconstruction_nrmse']>0
 np.savez(ev/'ground_truth.npz',bad=np.ones(1));report=judge.evaluate(ev/'baseline_reference.npz',ev);assert report['status']=='error' and report['metrics']=={}
def test_poor_valid_output(tmp_path):
 ev=evaluator(tmp_path);np.savez(tmp_path/'zero.npz',reconstruction=np.zeros((1,32,32)),density_sigma=np.zeros((1,32,32)));assert judge.evaluate(tmp_path/'zero.npz',ev)['status']=='fail'
def test_known_metrics():
 x=np.array([1.,2.,3.]);c,e=judge.reference_metrics(x,x);assert abs(c-1)<1e-15 and e==0
 assert judge.reference_metrics(np.zeros(3),x)[0]==0

@pytest.mark.parametrize('name,value',[('ncc_boundary',1.1),('ncc_boundary',-1.1),('ncc_boundary',True),('ncc_boundary','0.9')])
def test_invalid_ncc_configuration(tmp_path,name,value):
 ev=evaluator(tmp_path);config=json.loads((ev/'metrics.json').read_text());config['quantities']['density_sigma'][name]=value;(ev/'metrics.json').write_text(json.dumps(config));assert judge.evaluate(ev/'baseline_reference.npz',ev)['status']=='error'
def test_missing_evaluator_reference(tmp_path):
 ev=evaluator(tmp_path);(ev/'baseline_reference.npz').unlink();assert judge.evaluate(ROOT/'output/reconstruction.npz',ev)['status']=='error'
def test_foreign_cwd_cli_and_write_error(tmp_path):
 command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
 good=subprocess.run(command,cwd=tmp_path,capture_output=True);assert good.returncode==0;assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
 blocked=tmp_path/'blocked';blocked.write_text('file');command[-1]=str(blocked/'report.json');bad=subprocess.run(command,cwd=tmp_path,capture_output=True);assert bad.returncode==2 and b'Report write failure' in bad.stderr
def test_conditional_sampler_known_scalar():
 from scipy import sparse
 d=dict(B=sparse.eye(2),C=sparse.csr_matrix((2,1)),D=sparse.eye(2),weights=np.ones(2),normal=sparse.eye(2))
 samples=conditional_density_samples(d,np.array([2.,3.]),n_draw=10000,seed=9)
 np.testing.assert_allclose(samples.std(axis=1,ddof=1),[2.,3.],rtol=.03)
 with pytest.raises(ValueError):conditional_density_samples(d,np.ones(2),n_draw=1)
