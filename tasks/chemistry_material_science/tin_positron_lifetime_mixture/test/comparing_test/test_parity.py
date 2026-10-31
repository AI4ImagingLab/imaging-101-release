from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.physics_model import prediction_native,residual,package
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_integer_binning_and_original_forward():
 counts,prompt,y=load_data(ROOT/'data/raw_data.npz');area,bg=prepare(y);a=np.load(ROOT/'test/fixtures/upstream.npz');assert np.array_equal(y.sum(1),counts.sum(1))
 for i in range(6):
  p=np.array([198/50,.9,380/50,.1,2200/50,.005,198/50,5468/50,.5,198/50,5468/50,.5,bg[i]])
  np.testing.assert_allclose(prediction_native(p,area[i]),a['output_initial_prediction'][i],rtol=1e-10,atol=2e-8)
  r=residual(p,y[i],area[i]);np.testing.assert_allclose(r[:-1],a['output_initial_residual'][i,:914],rtol=1e-9,atol=2e-9);np.testing.assert_allclose(r[-1],a['output_initial_residual'][i,914],atol=1e-13)
  np.testing.assert_allclose(prediction_native(a['output_parameters_native'][i],area[i]),a['output_prediction'][i],rtol=1e-9,atol=3e-8)
  np.testing.assert_allclose(np.sum(residual(a['output_parameters_native'][i],y[i],area[i])**2),a['output_objective'][i],rtol=1e-11)

def test_zero_width_limit_and_scale_linearity():
 p=np.array([4.,1.,8.,0.,40.,0.,1e-8,100.,1.,4.,100.,0.,0.]);v=prediction_native(p,1.);delta=np.arange(1000)-33-100;ref=np.exp(-delta[delta>0]/4)-np.exp(-(delta[delta>0]+1)/4);np.testing.assert_allclose(v[delta>0],ref,rtol=1e-13,atol=1e-15);np.testing.assert_allclose(v[delta<-1],0,atol=1e-15);np.testing.assert_allclose(prediction_native(p,7),7*v,rtol=1e-15,atol=1e-15)

def test_all_six_complete_local_conditions():
 _,_,y=load_data(ROOT/'data/raw_data.npz');p,v,h=reconstruct(y);a=np.load(ROOT/'test/fixtures/upstream.npz');local_contract=package(p,v);reference_contract=package(a['output_parameters_native'],a['output_prediction']);
 for key in local_contract:
  if key!='prediction':np.testing.assert_allclose(local_contract[key],reference_contract[key],rtol=1e-4,atol=.01)
 np.testing.assert_allclose(v,a['output_prediction'],rtol=3e-5,atol=.01);assert len(h)==6 and all(len(x)<=20 for x in h)
 with np.load(ROOT/'output/reconstruction.npz') as current:
  for k,value in package(p,v).items():np.testing.assert_array_equal(current[k],value)
