from pathlib import Path
import sys,numpy as np,torch
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T));torch.set_num_threads(1)
from src.physics_model import GP
from src.preprocessing import load,prepare
MAP={'task_noise':'likelihood.raw_task_noises','global_noise':'likelihood.raw_noise','factor':'covar_module.task_covar_module.covar_factor','task_var':'covar_module.task_covar_module.raw_var','length':'covar_module.data_covar_module.raw_lengthscale','mean0':'mean_module.base_means.0.raw_constant','mean1':'mean_module.base_means.1.raw_constant'}
def inject(m,f):
 with torch.no_grad():
  for k,v in MAP.items():m.p[k].copy_(torch.tensor(f['param_'+v]))
def test_original_initial_covariance_loss_gradient_adam():
 with np.load(T/'test/fixtures/upstream_gp_stages.npz') as f:
  m=GP();inject(m,f);x=torch.tensor(f['input_train']);y=torch.tensor(f['input_locations']);opt=torch.optim.Adam(m.parameters(),lr=.1)
  np.testing.assert_allclose(m.cov(x).detach(),f['output_initial_covariance'],rtol=3e-7,atol=5e-7)
  loss=m.nll(x,y);np.testing.assert_allclose(loss.detach(),f['output_initial_negative_log_likelihood'],rtol=3e-7);loss.backward()
  for k,v in MAP.items():np.testing.assert_allclose(m.p[k].grad,f['output_gradient_'+v],rtol=4e-6,atol=2e-5)
  opt.step()
  for k,v in MAP.items():np.testing.assert_allclose(m.p[k].detach(),f['output_update_'+v],rtol=2e-6,atol=3e-8)
def test_original_fixed_posterior_and_preprocessing():
 with np.load(T/'test/fixtures/upstream_gp_posterior.npz') as f:
  m=GP();inject(m,f);x,y,z=prepare(load(T/'data/raw_data.npz'))
  np.testing.assert_array_equal(x,f['input_train']);np.testing.assert_array_equal(y,f['input_locations']);np.testing.assert_array_equal(z,f['input_test'])
  with torch.no_grad():p,c=m.predict(x,y,z)
  np.testing.assert_allclose(p,f['output_mean'],rtol=8e-6,atol=1e-4);np.testing.assert_allclose(c,f['output_covariance'],rtol=2e-4,atol=.003)
def test_full_fresh_result_physical_accuracy():
 with np.load(T/'output/reconstruction.npz') as f:p=f['reconstruction']
 with np.load(T/'evaluation/ground_truth.npz') as f:y=f['reconstruction']
 errors=np.linalg.norm(p-y,axis=-1);assert np.all(np.sqrt(np.mean(errors**2,axis=-1))<np.array([[9.91144171,10.541507,11.1028762]]))
