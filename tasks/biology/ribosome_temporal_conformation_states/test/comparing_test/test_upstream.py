import numpy as np
from src.physics_model import variational_step
def test_original_e_step(root):
 with np.load(root/'test/fixtures/upstream_stage.npz') as z:
  x=z['observations'];r,xi,ll=variational_step(x[None],np.array([len(x)]),z['a'],z['b'],z['beta'],z['means'],z['pi'],z['transition_posterior']);np.testing.assert_allclose(r[0],z['responsibilities'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(xi,z['transition_responsibilities'].sum(0),atol=1e-10,rtol=1e-12);np.testing.assert_allclose(ll,z['log_evidence'],atol=1e-10)
def test_full_pipeline_all_quantities(root,fresh,judge):
 p,fit=fresh;report,code=judge.evaluate(p);assert code==0,report
 with np.load(root/'evaluation/baseline_reference.npz') as z:
  for key in ['state_mean','state_variance','transition_probability','initial_probability','population_fraction']:np.testing.assert_allclose(fit[key],z[key][0],rtol=1e-11,atol=1e-12)
  np.testing.assert_array_equal(fit['reconstruction'],z['reconstruction'][0])
 assert fit['posterior'].shape==(64281,2)
 # Each sequence contributes exactly one start and no cross-sequence transition.
 np.testing.assert_allclose(fit['pi'].sum(),590+2,atol=1e-10);np.testing.assert_allclose(fit['transition_posterior'].sum(),64281-590+4,atol=1e-8)
def test_original_posterior_updates(root,fresh):
 with np.load(root/'test/fixtures/upstream_posterior.npz') as z:
  fit=fresh[1]
  for key in ['a','b','beta']:np.testing.assert_allclose(fit[key],z[key],atol=1e-9,rtol=1e-11)
