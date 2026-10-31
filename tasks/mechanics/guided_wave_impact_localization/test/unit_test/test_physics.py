from pathlib import Path
import sys,numpy as np,torch
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T));torch.set_num_threads(1)
from src.physics_model import GP
from src.preprocessing import load,prepare

def test_one_point_two_independent_scalar_conditionals():
 m=GP(dtype=torch.float64)
 with torch.no_grad():m.p['factor'].zero_();m.p['mean0'].fill_(2);m.p['mean1'].fill_(-3)
 x=torch.tensor([[0.,0.]],dtype=torch.float64);y=torch.tensor([[7.,4.]],dtype=torch.float64);z=torch.tensor([[.4,.1]],dtype=torch.float64)
 p,c=m.predict(x,y,z);v=torch.diag(m.task()).detach().numpy();noise=m.noise().detach().numpy();rho=np.exp(-.5*(.4**2+.1**2)/float(torch.nn.functional.softplus(m.p['length']).detach())**2)
 expected=np.array([2.,-3.])+rho*v/(v+noise)*(np.array([7.,4.])-np.array([2.,-3.]));np.testing.assert_allclose(p.detach()[0],expected,rtol=1e-13)
 np.testing.assert_allclose(np.diag(c.detach()),v+noise-rho**2*v**2/(v+noise),rtol=1e-13);assert abs(float(c.detach()[0,1]))<1e-15

def test_float64_likelihood_gradient_and_covariance_psd():
 torch.manual_seed(8);m=GP(dtype=torch.float64);x=torch.tensor([[0.,0.],[.2,.5],[.9,.1]],dtype=torch.float64);y=torch.tensor([[1.,3.],[4.,2.],[7.,5.]],dtype=torch.float64);loss=m.nll(x,y);loss.backward()
 for key in ('length','factor','task_noise','global_noise','task_var','mean0','mean1'):
  p=m.p[key];analytic=float(p.grad.flatten()[0]);old=float(p.detach().flatten()[0]);h=1e-5
  with torch.no_grad():p.flatten()[0]=old+h
  plus=float(m.nll(x,y).detach())
  with torch.no_grad():p.flatten()[0]=old-h
  minus=float(m.nll(x,y).detach())
  with torch.no_grad():p.flatten()[0]=old
  np.testing.assert_allclose(analytic,(plus-minus)/(2*h),rtol=2e-7,atol=2e-8)
 _,c=m.predict(x,y,x);assert np.linalg.eigvalsh(c.detach().numpy()).min()>0

def test_native_calibration_units_reference_and_scale_invariance():
 d=load(T/'data/raw_data.npz');assert d['calibration_tdoa'].dtype==np.float64;assert d['observations'].dtype==np.float64
 np.testing.assert_array_equal(d['sensor_positions'],[[0,0],[80,0],[160,0],[0,120],[80,120],[160,120]])
 assert np.all(d['observations'].min(-1)==0);assert np.all(d['calibration_tdoa'].min(-1)==0);assert 1e-6<d['observations'].max()<1e-3
 x,y,z=prepare(d);other={**d,'observations':d['observations']*2};np.testing.assert_array_equal(prepare(other)[2],z)
 other={**d,'observations':d['observations']*1.3};np.testing.assert_allclose(prepare(other)[2],z,rtol=2e-7,atol=1e-7)
 readme=(T/'README.md').read_text(encoding='utf8');assert '2 | MHz' in readme and '290 x 200 x 4' in readme
