from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare
from src.physics_model import initialize_variance,expectation,transform
from src.solvers import one_step,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_preprocessing_and_original_initial_variance():
 f=np.load(ROOT/'test/fixtures/upstream_stages.npz');d=load_data(ROOT/'data/raw_data.npz');x,y,origin,unit=prepare(d)
 np.testing.assert_array_equal(x,f['input_target_normalized']);np.testing.assert_array_equal(y,f['input_source_normalized'])
 np.testing.assert_array_equal(origin,f['config_origin']);assert unit==f['config_coordinate_unit_m']
 assert initialize_variance(x,y)==f['output_initial_sigma2']

def test_original_e_step_and_first_m_step():
 f=np.load(ROOT/'test/fixtures/upstream_stages.npz');x=f['input_target_normalized'];y=f['input_source_normalized'];sigma=float(f['output_initial_sigma2'])
 e=expectation(x,y,sigma,.1)
 for k,v in zip(['P','P1','Pt1','PX','Np'],e):np.testing.assert_allclose(v,f['output_initial_'+k],rtol=1e-10,atol=1e-11)
 step=one_step(x,y,y,sigma,np.inf,.1)
 for k in ['R','t','s','TY','sigma2','A','YPY','q']:np.testing.assert_allclose(step[k],f['output_first_'+k],rtol=1e-10,atol=1e-11)

def test_full_native_pipeline_and_history():
 result,info=reconstruct(load_data(ROOT/'data/raw_data.npz'));f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
 with np.load(ROOT/'evaluation/baseline_reference.npz') as ref:
  for k in result:np.testing.assert_allclose(result[k],ref[k],rtol=1e-10,atol=1e-11)
 np.testing.assert_allclose(info['history'],f['output_history'],rtol=1e-9,atol=1e-8)
