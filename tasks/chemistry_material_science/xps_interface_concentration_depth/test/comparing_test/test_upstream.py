from pathlib import Path
import numpy as np,pytest
from src.preprocessing import load_data,mean_observation
from src.physics_model import second_difference,objective
from src.solvers import reconstruct,dual_prox
F=Path(__file__).resolve().parents[1]/'fixtures/upstream.npz'
@pytest.fixture(scope='module')
def computed():
 d=load_data();return reconstruct(d['system_matrix'],d['observations'])
def test_exact_bound_observations():
 d=load_data()
 with np.load(F) as f:
  np.testing.assert_array_equal(d['system_matrix'],f['H']);np.testing.assert_array_equal(d['observations'],f['observations'])
  np.testing.assert_allclose(mean_observation(d['observations']),f['mean'],atol=2e-14,rtol=2e-14)
def test_original_operator():
 with np.load(F) as f:np.testing.assert_array_equal(second_difference(201),f['D'])
def test_original_proximal_and_cost():
 with np.load(F) as f:
  target=np.r_[f['mean'],np.zeros(199)];cov=np.r_[np.diag(f['Gamma_y']),np.diag(f['Gamma_d'])]
  np.testing.assert_allclose(dual_prox(np.linspace(-.2,.8,209),target,.3,cov),f['prox'],atol=2e-9,rtol=2e-9)
  cost=objective(f['reconstruction'],f['H'],f['mean'],.01**2/199,201/10*.01**2)
  np.testing.assert_allclose(cost,f['objective'][0],atol=2e-9,rtol=2e-9)
def test_original_first_50_nonuniform_updates(computed):
 x,d=computed
 with np.load(F) as f:
  for a,k in [('primal_path','X_path'),('dual_path','S_path'),('tau_path','tau_path')]:np.testing.assert_allclose(d[a],f[k],rtol=2e-9,atol=2e-9)
  assert np.ptp(f['X_path'][:,-1])>.1
def test_original_full_pipeline(computed):
 x,d=computed
 with np.load(F) as f:np.testing.assert_allclose(x,f['reconstruction'],atol=2e-7,rtol=2e-6)
 assert d['iterations']==100000
def test_entrypoint_and_truth_quality(tmp_path):
 from src.main import run
 from evaluation.judge import evaluate
 p=tmp_path/'submission.npz';x,d,seconds=run(p);r,c=evaluate(p);assert c==0 and r['passed'];assert seconds>0
