import numpy as np
from pathlib import Path
from src.physics_model import internal_states
P=Path(__file__).resolve().parents[1]/'fixtures'
def test_full_original_internal_states():
 with np.load(P/'upstream_states.npz') as f:
  a,k,s=internal_states(f['parameters'],f['plastic_strain'])
  for got,key in [(a,'output_backstress'),(k,'output_isotropic'),(s,'output_stress')]:assert np.allclose(got,f[key],rtol=2e-12,atol=1e-10)
def test_eight_frozen_original_parameter_models():
 with np.load(P/'upstream_initials.npz') as f,np.load(P/'upstream_states.npz') as states:
  for q,s in zip(f['parameters'],f['predictions']):
   assert np.allclose(internal_states(q,states['plastic_strain'])[2],s,rtol=2e-12,atol=1e-10)
