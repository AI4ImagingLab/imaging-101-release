from pathlib import Path
import numpy as np,pytest,json,sys,subprocess
from src.preprocessing import load_data
from src.physics_model import density_basis,derivative_basis
from src.solvers import objective,derivatives,newton_direction
from src.generate_data import ray_geometry,clean_measurements,generate
from evaluation.judge import pair_metrics,evaluate
T=Path(__file__).resolve().parents[2]
def test_independent_slab_geometry():
 data=load_data();e,x,n,l=ray_geometry();assert len(e)==1798
 np.testing.assert_allclose(e,data['entry'],atol=1e-13,rtol=0);np.testing.assert_allclose(x,data['exit'],atol=1e-13,rtol=0);np.testing.assert_allclose(n*l[:,None],x-e,atol=1e-15)
def test_author_noise_and_independent_forward_physics():
 d=load_data();clean=clean_measurements(d['entry'],d['exit'],d['direction']);noise=(d['observations']-clean)/4.056e-4
 assert abs(noise.mean())<.1 and .9<noise.std(ddof=1)<1.1
def test_independent_generator_complete_acquisition_contract():
 data=load_data();new=generate();assert set(new)==set(data)
 for key in data:assert new[key].shape==data[key].shape and new[key].dtype==np.float64 and np.isfinite(new[key]).all()
 clean=clean_measurements(new['entry'],new['exit'],new['direction']);noise=(new['observations']-clean)/4.056e-4
 assert abs(noise.mean())<.1 and .9<noise.std(ddof=1)<1.1
def test_original_analytic_integrals_against_independent_quadrature():
 d=load_data()
 with np.load(T/'test/fixtures/stages.npz',allow_pickle=False) as f:
  b={k:f[k].ravel() if f[k].size>1 else float(f[k].item()) for k in ['ldx','ldy','lcx','lcy','Ldx','Ldy','Lcx','Lcy']};ids=f['ids'].ravel().astype(int)-1;nodes,w=np.polynomial.legendre.leggauss(64);nodes=(nodes+1)/2;w=w/2;m=len(b['ldx'])
  for i in [0,3,6,10]:
   ray=ids[i];pts=d['entry'][ray]+nodes[:,None]*(d['exit'][ray]-d['entry'][ray]);D=density_basis(pts,b);C=derivative_basis(pts,b)
   for j,k in [(0,0),(8,13),(m-1,len(b['lcx'])-1)]:
    for c in range(3):
     integral=d['path_length'][ray]*np.sum(w*D[:,j]*C[:,k,c]);assert abs(integral-f['ip'][i,j+m*k,c])/max(1,abs(integral))<2e-8
def test_objective_gradient_and_negative_gradient_hessian():
 d=load_data()
 with np.load(T/'test/fixtures/stages.npz',allow_pickle=False) as f,np.load(T/'test/fixtures/trace.npz',allow_pickle=False) as tr:
  ids=f['ids'].ravel().astype(int)-1;ip=f['ip'];v=d['direction'][ids];length=d['path_length'][ids];m=f['ldx'].size
  B=(v[:,0,None]**2*(ip[:,:,1]-.3*ip[:,:,0])-2*1.3*(v[:,0]*v[:,1])[:,None]*ip[:,:,2]+v[:,1,None]**2*(ip[:,:,0]-.3*ip[:,:,1]))/length[:,None];B=B.reshape(len(ids),m,-1,order='F');A=f['id0']/length[:,None];tb=f['tb'];mean=tr['mu'].ravel();prior=tr['SLambda'].ravel();x=tr['what'][:,1];scale=tr['coordScale'].ravel();y=d['observations'][ids];g,H,J=derivatives(x,mean,prior,A,B,tb,y)
  vector=np.random.default_rng(55).standard_normal(len(x));vector/=np.linalg.norm(vector);p=scale*vector;eps=1e-3
  fd=(objective(x+eps*p,mean,prior,A,B,tb,y)-objective(x-eps*p,mean,prior,A,B,tb,y))/(2*eps);exact=-g@p;assert abs(fd-exact)/max(1,abs(exact))<2e-5
  gp=derivatives(x+eps*p,mean,prior,A,B,tb,y)[0];gm=derivatives(x-eps*p,mean,prior,A,B,tb,y)[0];fd=scale*(gp-gm)/(2*eps);exact=-scale*(H@p);assert np.linalg.norm(fd-exact)/max(1,np.linalg.norm(exact))<2e-5
  initial1=derivatives(mean,mean,prior,A,B,tb,y)[1];initial2=derivatives(mean,mean,prior,A,B,tb,y+1.)[1]
  np.testing.assert_array_equal(np.diag(initial1),np.diag(initial2))
def test_coefficient_unit_change_congruence_invariance():
 with np.load(T/'test/fixtures/trace.npz',allow_pickle=False) as f:
  g=f['first_gradient'].ravel();h=f['first_hessian'];s=f['coordScale'].ravel();m=f['lambda_d0x'].size;units=np.r_[np.full(m,2.**7),np.full(len(g)-m,2.**-9)]
  step,_,rank=newton_direction(g,h,s);converted,_,rank2=newton_direction(units*g,units[:,None]*h*units[None,:],s/units)
  np.testing.assert_allclose(units*converted,step,rtol=1e-12,atol=1e-15);assert rank==rank2==188
def test_known_metrics():
 n,e=pair_metrics(np.array([0.,1.,1.]),np.array([0.,1.,2.]));assert abs(n-3/np.sqrt(10))<1e-15;assert abs(e-np.sqrt(1/3)/2)<1e-15
@pytest.mark.parametrize('kind',['missing','key','shape','float32','complex','nan','inf'])
def test_malformed_submission(tmp_path,kind):
 p=tmp_path/'bad.npz';a={'strain':np.zeros((1,3,40,40),np.float64),'lattice_spacing':np.zeros((1,40,40),np.float64)}
 if kind=='key':a.pop('lattice_spacing')
 if kind=='shape':a['strain']=a['strain'][0]
 if kind=='float32':a['strain']=a['strain'].astype(np.float32)
 if kind=='complex':a['strain']=a['strain'].astype(complex)
 if kind=='nan':a['strain'].flat[0]=np.nan
 if kind=='inf':a['strain'].flat[0]=np.inf
 if kind!='missing':np.savez(p,**a)
 r,c=evaluate(p);assert c==2 and r['metrics']=={} and r['errors']
@pytest.mark.parametrize('kind',['zero','constant_spacing','zero_one_strain'])
def test_poor_or_incomplete_physical_fields_fail(tmp_path,kind):
 p=tmp_path/'poor.npz'
 with np.load(T/'evaluation/ground_truth.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 if kind=='zero':a={k:np.zeros_like(v) for k,v in a.items()}
 if kind=='constant_spacing':a['lattice_spacing'][:]=4.056
 if kind=='zero_one_strain':a['strain'][:,1]=0
 np.savez(p,**a);r,c=evaluate(p);assert c==1 and not r['passed']
def test_foreign_cwd_and_report(tmp_path):
 report=tmp_path/'nested/report.json';r=subprocess.run([sys.executable,str(T/'evaluation/judge.py'),'--prediction',str(T/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0;assert json.loads(report.read_text())['passed']
def test_evaluator_and_report_write_failures(tmp_path):
 import evaluation.judge as judge
 old=judge.ASSETS;judge.ASSETS=tmp_path
 try:r,c=judge.evaluate(T/'output/reconstruction.npz');assert c==2 and r['metrics']=={}
 finally:judge.ASSETS=old
 block=tmp_path/'file';block.write_text('blocked');assert judge.main(['--prediction',str(T/'output/reconstruction.npz'),'--report',str(block/'report.json')])==2
def test_public_constants_keys_and_privacy():
 text=(T/'README.md').read_text(encoding='utf8');d=load_data()
 for key in d:assert '| '+key+' |' in text
 for term in ['github.com','ground_truth.npz','2000','1203','0.0168']:assert term not in text
 assert '0.0004056' in text and '4.056' in text and '0.3' in text and d['grid_x'].shape==(40,40)
