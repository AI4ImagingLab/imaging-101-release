from pathlib import Path
import json,subprocess,sys,numpy as np,pytest
from src.preprocessing import load_data
from src.physics_model import forward,adjoint,second_difference,objective
from src.solvers import dual_prox
from src.visualization import metrics
from evaluation.judge import evaluate,score
T=Path(__file__).resolve().parents[2]
def test_forward_adjoint_and_units():
 d=load_data();h=d['system_matrix'];x=np.linspace(.2,1.2,201);v=np.linspace(-1,2,10)
 assert abs(forward(x,h)@v-x@adjoint(v,h))<1e-12
 np.testing.assert_allclose(forward(2*x,h),2*forward(x,h),rtol=0,atol=0)
 assert np.all(h>0) and np.all(np.diff(h,axis=0)>=-1e-12)
def test_depth_and_boundary_convention():
 d=load_data();np.testing.assert_allclose(d['depth_nm'],np.linspace(-2,20,201),atol=4e-12)
 a=second_difference(201);np.testing.assert_allclose(a@np.ones(201),0,atol=0);np.testing.assert_allclose(a@np.arange(201),0,atol=0)
 assert a.shape==(199,201) and np.all(a[:,0][1:]==0) and a[0,0]==-1
def test_independent_prox_stationarity():
 v=np.linspace(-2,3,17);b=np.linspace(.1,.9,17);cov=np.linspace(.001,.5,17);sigma=.7
 u=dual_prox(v,b,sigma,cov);np.testing.assert_allclose(u-v+sigma*(b+.5*cov*u),0,atol=1e-15)
def test_objective_directional_derivative():
 data=load_data();h=data['system_matrix'];y=data['observations'].mean(0);d=second_difference(201);x=np.linspace(.1,1.1,201);v=np.sin(np.arange(201));vy=.01**2/199;vd=201/10*.01**2
 grad=2*h.T@(h@x-y)/vy+2*d.T@(d@x)/vd;eps=1e-6
 fd=(objective(x+eps*v,h,y,vy,vd)-objective(x-eps*v,h,y,vy,vd))/(2*eps)
 np.testing.assert_allclose(fd,grad@v,rtol=1e-7,atol=1e-5)
def test_metric_known_case():
 r=np.array([0.,1.,2.]);x=np.array([0.,1.,1.]);m=score(x,r);assert abs(m['ncc']-3/np.sqrt(10))<1e-15;assert abs(m['nrmse']-np.sqrt(1/3)/2)<1e-15
 np.testing.assert_allclose(metrics(x,r),list(m.values()),rtol=1e-15)
@pytest.mark.parametrize('kind',['missing_key','shape','dtype','nan','inf','complex','missing'])
def test_malformed_predictions(tmp_path,kind):
 p=tmp_path/'bad.npz';a=np.ones((1,201),np.float64)
 if kind=='missing':pass
 elif kind=='missing_key':np.savez(p,other=a)
 else:
  if kind=='shape':a=a[0]
  if kind=='dtype':a=a.astype(np.float32)
  if kind=='complex':a=a.astype(complex)
  if kind in ['nan','inf']:a.flat[0]=np.nan if kind=='nan' else np.inf
  np.savez(p,reconstruction=a)
 report,c=evaluate(p);assert c==2 and report['metrics']=={} and report['errors']
@pytest.mark.parametrize('level',[0.,1.])
def test_poor_constant_fails(tmp_path,level):
 p=tmp_path/'poor.npz';np.savez(p,reconstruction=np.full((1,201),level,dtype=np.float64));r,c=evaluate(p);assert c==1 and not r['passed']
def test_foreign_cwd_cli(tmp_path):
 p=T/'output/reconstruction.npz';r=tmp_path/'nested/report.json';process=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(p),'--report',str(r)],cwd=tmp_path,capture_output=True,text=True)
 assert process.returncode==0;assert json.loads(r.read_text())['passed']
def test_unavailable_assets_and_write_failure(tmp_path):
 import evaluation.judge as judge
 old=judge.ASSETS;judge.ASSETS=tmp_path
 try:r,c=judge.evaluate(T/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
 finally:judge.ASSETS=old
 block=tmp_path/'file';block.write_text('blocked');assert judge.main(['--prediction',str(T/'output/reconstruction.npz'),'--report',str(block/'report.json')])==2
def test_public_completeness_and_privacy():
 text=(T/'README.md').read_text(encoding='utf8');d=load_data()
 for k in d:assert '| '+k+' |' in text
 for term in ['github.com','github.io','ground_truth.npz','concentration_profile.csv']:assert term not in text
 assert d['observations'].shape==(199,10) and d['observations'].min()>.01*20
