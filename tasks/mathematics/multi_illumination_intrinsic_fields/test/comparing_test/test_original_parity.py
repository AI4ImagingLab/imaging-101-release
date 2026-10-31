from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from src.preprocessing import load_observations
from src.physics_model import gradients,normal_equations
from src.solvers import estimate,integrate_gradients
T=Path(__file__).resolve().parents[2];F=T/'test/fixtures'
def test_original_median_targets_and_matrix():
 x=load_observations(T/'data/raw_data.npz');gy,gx=gradients(np.log(np.maximum(x['multiple'],3.0)))
 with np.load(F/'original_primary_stages.npz') as z:
  ty=np.median(gy,axis=-1);tx=np.median(gx,axis=-1)
  np.testing.assert_allclose(ty,z['target_y'],rtol=0,atol=1e-12);np.testing.assert_allclose(tx,z['target_x'],rtol=0,atol=1e-12)
  A,b,numbers=normal_equations(ty,tx,x['mask']);np.testing.assert_allclose(b,z['b'],rtol=0,atol=1e-12);np.testing.assert_array_equal(numbers,z['numbers'])
  delta=A-load_npz(F/'original_masked_laplacian.npz');assert delta.nnz==0 or np.max(abs(delta.data))<1e-12
  with np.load(F/'primary.npz') as primary:np.testing.assert_array_equal(np.exp(z['integrated'][x['mask']]),primary['reflectance'][x['mask']])
def test_full_pipeline_source_accuracy_envelope():
 x=load_observations(T/'data/raw_data.npz');out=estimate(x['image'],x['multiple'],x['mask']);m=x['mask']
 for filename,boundary in [('high_accuracy_primary.npz',1e-7),('primary.npz',.001)]:
  with np.load(F/filename) as z:
   scale=np.exp(np.log(z['reflectance'][m]).mean())
   for key,factor in [('shading',scale),('reflectance',1/scale)]:
    ref=z[key]*factor;error=np.sqrt(np.mean((out[key][m]-ref[m])**2))/np.ptp(ref[m]);assert error<=boundary,(key,filename,error)
    if filename=='primary.npz':assert np.dot(out[key][m],ref[m])/np.linalg.norm(out[key][m])/np.linalg.norm(ref[m])>=.99999
def test_analytic_independent_median_recovery():
 with np.load(F/'analytic_median_and_laplacian.npz') as z:
  out=estimate(z['input'],z['lights'],np.ones(z['input'].shape,bool))
  np.testing.assert_allclose(out['reflectance'],z['true_reflectance'],rtol=0,atol=1e-8)
  np.testing.assert_allclose(out['shading'],z['true_shading'],rtol=0,atol=1e-6)
  ty,tx=gradients(np.log(z['true_reflectance']));A,b,_=normal_equations(ty,tx,np.ones(z['input'].shape,bool))
  np.testing.assert_array_equal(A.toarray(),z['matrix']);np.testing.assert_allclose(b,z['rhs'],atol=1e-14,rtol=0)
  np.testing.assert_allclose(np.linalg.eigvalsh(A.toarray()),z['eigenvalues'],atol=1e-14,rtol=0)
def test_independent_original_repeat_and_gauge_capture():
 x=load_observations(T/'data/raw_data.npz');m=x['mask']
 with np.load(F/'primary.npz') as first,np.load(F/'repeat.npz') as repeat:
  for key in ('shading','reflectance'):np.testing.assert_array_equal(first[key],repeat[key])
 with np.load(F/'high_accuracy_primary.npz') as z:
  gauge=np.exp(np.log(z['raw_reflectance'][m]).mean())
  np.testing.assert_array_equal(z['raw_reflectance']/gauge,z['reflectance'])
  np.testing.assert_array_equal(z['raw_shading']*gauge,z['shading'])
