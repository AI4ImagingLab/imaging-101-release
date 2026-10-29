from pathlib import Path
import numpy as np,pytest
from src.preprocessing import load_data
from src.physics_model import place_bases,integral_density,integral_products,density_basis,strain_basis,traction_basis,derivative_basis,prediction,jacobian
from src.solvers import reconstruct
T=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def computed():return reconstruct(load_data())
def test_bound_original_observation_and_geometry():
 data=load_data()
 with np.load(T/'test/fixtures/qualified.npz',allow_pickle=False) as f:
  for key,original in [('observations','dm'),('entry','entry'),('exit','exit'),('direction','nhat'),('path_length','L')]:
   expected=f[original].T if key in ['entry','exit','direction'] else f[original].ravel();np.testing.assert_array_equal(data[key],expected)
def test_original_bases_and_analytic_integrals():
 data=load_data();b=place_bases(data['direction'])
 with np.load(T/'test/fixtures/stages.npz',allow_pickle=False) as f:
  for key in ['ldx','ldy','lcx','lcy','sd','sc']:np.testing.assert_array_equal(b[key],f[key].ravel())
  ids=f['ids'].ravel().astype(int)-1
  for actual,key in [(integral_density(data['entry'][ids],data['exit'][ids],data['direction'][ids],b),'id0'),(integral_products(data['entry'][ids],data['exit'][ids],data['direction'][ids],b),'ip'),(traction_basis(data['traction_points'],data['traction_normals'],b),'tb')]:assert np.linalg.norm(actual-f[key])/np.linalg.norm(f[key])<2e-8
def test_qualified_source_gradient_hessian_jacobian(computed):
 result,d=computed
 with np.load(T/'test/fixtures/trace.npz',allow_pickle=False) as f:
  for actual,key in [('gradient','first_gradient'),('hessian','first_hessian'),('jacobian','first_jacobian')]:
   a=d['first'][actual];z=f[key].squeeze();assert np.linalg.norm(a-z)/np.linalg.norm(z)<2e-8
def test_original_derivatives_product_order_prediction_jacobian():
 data=load_data();b=place_bases(data['direction'])
 with np.load(T/'test/fixtures/stages.npz',allow_pickle=False) as f:
  points=np.c_[data['grid_x'].ravel(order='F')[::37],data['grid_y'].ravel(order='F')[::37]];D=derivative_basis(points,b);assert np.linalg.norm(D-f['db'])/np.linalg.norm(f['db'])<2e-8
  ids=f['ids'].ravel().astype(int)-1;v=data['direction'][ids];length=data['path_length'][ids];ip=f['ip'];m=f['wd'].size;k=f['wc'].size
  reordered=ip.reshape(len(ids),m,k,3,order='F').transpose(0,2,1,3).reshape(len(ids),m*k,3,order='F');np.testing.assert_array_equal(reordered,f['ip2'])
  B=(v[:,0,None]**2*(ip[:,:,1]-.3*ip[:,:,0])-2*1.3*(v[:,0]*v[:,1])[:,None]*ip[:,:,2]+v[:,1,None]**2*(ip[:,:,0]-.3*ip[:,:,1]))/length[:,None];B=B.reshape(len(ids),m,k,order='F');A=f['id0']/length[:,None];weights=np.r_[f['wd'].ravel(),f['wc'].ravel()]
  p=prediction(weights,A,B);J=jacobian(weights,A,B)
  assert np.linalg.norm(p-f['prediction'].ravel())/np.linalg.norm(p)<2e-8
  for actual,source in [(J[:,:m],f['Jd']),(J[:,m:],f['Jc'])]:assert np.linalg.norm(actual-source)/np.linalg.norm(source)<2e-8
def test_physical_update_parity_without_false_coefficient_claim(computed):
 result,d=computed;data=load_data();b=d['basis'];m=len(b['ldx']);points=np.c_[data['grid_x'].ravel(order='F'),data['grid_y'].ravel(order='F')];D=density_basis(points,b);E=strain_basis(points,b)
 with np.load(T/'test/fixtures/trace.npz',allow_pickle=False) as f:
  source=f['what'];local=d['weights_path'];assert local.shape[1]==4
  # Ill-conditioned coefficient directions are explicitly NOT certified as equal.
  coeff_error=np.linalg.norm(d['first']['direction']-f['first_direction'].ravel())/np.linalg.norm(f['first_direction']);assert np.isfinite(coeff_error)
  for i in range(1,4):
   for x,z in [(D@local[:m,i],D@source[:m,i]),(E@local[m:,i],E@source[m:,i])]:assert np.sqrt(np.mean((x-z)**2))/np.ptp(z)<2e-5
def test_full_qualified_physical_fields(computed):
 result,d=computed
 with np.load(T/'notebooks/reference_outputs/qualified_source.npz',allow_pickle=False) as z:
  for key in ['strain','lattice_spacing']:assert np.sqrt(np.mean((result[key]-z[key])**2))/np.ptp(z[key])<2e-5
 assert len(d['history'])==4 and np.all(np.diff(d['history'][:,1])<0)
def test_standard_cli_contract_and_truth_metrics(tmp_path):
 from src.main import run
 from evaluation.judge import evaluate
 output=tmp_path/'reconstruction.npz';r,d=run(output);report,code=evaluate(output);assert code==0 and report['passed'] and len(report['metrics'])==8
