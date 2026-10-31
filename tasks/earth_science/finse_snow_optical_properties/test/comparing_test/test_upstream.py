from pathlib import Path
import sys
import numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load
from src.physics_model import forward,step

def test_original_forward_gradient_update():
 d=load(T/'data/raw_data.npz')
 with np.load(T/'test/fixtures/upstream_stages.npz') as f:
  
  for i in range(12):
   np.testing.assert_array_equal(d[f'weight_{i}'],f[f'weight_{i}']);np.testing.assert_array_equal(d[f'bias_{i}'],f[f'bias_{i}'])
  q=f['input_variables'];y=d['spectra'][:8,5::10][:,:100].astype(np.float32);z=forward(q,d);r,g=forward(q,d,y);qn,acc=step(q,g,np.full_like(q,.1))
  np.testing.assert_allclose(z,f['output_forward'],rtol=2e-6,atol=4e-7)
  np.testing.assert_allclose(g,f['output_gradient'],rtol=2e-5,atol=5e-6)
  np.testing.assert_allclose(qn,np.abs(f['output_update']),rtol=3e-6,atol=7e-7)

def test_original_same_parameter_full_field_repeat():
 d=load(T/'data/raw_data.npz')
 with np.load(T/'test/fixtures/upstream_final_forward.npz') as f:
  z=forward(f['input_normalized'],d)
  np.testing.assert_array_equal(f['output_model'],f['output_repeat'])
  np.testing.assert_allclose(z,f['output_model'],rtol=2e-6,atol=4e-7)
  y=np.tile(d['spectra'][:,5::10][:,:100],(2,1))
  local=np.mean(abs(z[:,6:106]-y),1);original=np.mean(abs(f['output_model'][:,6:106]-y),1)
  assert abs(local.mean()-original.mean())<1e-7
  assert abs(local.max()-original.max())<1e-6
