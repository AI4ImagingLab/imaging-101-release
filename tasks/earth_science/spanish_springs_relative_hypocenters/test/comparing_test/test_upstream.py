from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load,algorithm_inputs
from src.physics_model import relocate_pair
from src.solvers import quality_order

def test_every_original_preprocessed_input_and_quality_heap_order():
 d=algorithm_inputs(load(T/'data/raw_data.npz'))
 with np.load(T/'test/fixtures/upstream_inputs.npz') as f:
  for k,v in d.items():np.testing.assert_array_equal(v,f['output_'+k])
  quality,order=quality_order(d);np.testing.assert_array_equal(quality,f['output_quality']);np.testing.assert_array_equal(order,f['output_sort_index']-1)

def test_original_first_relative_grid_search_all_quantities():
 d=algorithm_inputs(load(T/'data/raw_data.npz'))
 with np.load(T/'test/fixtures/upstream_relative_search.npz') as f:
  g={k[6:]:f[k] for k in f.files if k.startswith('input_')};center=[f['param_centroid_lat'][0],f['param_centroid_lon'][0],f['param_centroid_depth'][0]];q,t,r,separation,rms,med,res=relocate_pair(center,g,d['travel_times'],d['distance_grid'],d['depth_grid']);expected=np.array([f['output_'+k][0] for k in ['latitude1','longitude1','depth1','latitude2','longitude2','depth2']]);np.testing.assert_array_equal(q,expected);np.testing.assert_array_equal(r,f['output_residual']);np.testing.assert_array_equal(np.array([t,separation,rms,med]),np.array([f['output_'+k][0] for k in ['origin_difference','separation','rms','median_absolute']]))

def test_full_original_and_actual_fresh_complete_catalog():
 with np.load(T/'output/reconstruction.npz') as f:q=f['reconstruction'][0]
 with np.load(T/'test/fixtures/upstream_final_catalog.npz') as f:expected=np.column_stack([f['output_'+k] for k in ['latitude','longitude','depth','origin_correction']])
 np.testing.assert_array_equal(q,expected)
