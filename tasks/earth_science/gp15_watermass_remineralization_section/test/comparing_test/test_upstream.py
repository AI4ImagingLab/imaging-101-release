import numpy as np
from src.preprocessing import load
from src.physics_model import usage_penalties,predict,objective_gradient
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'fixtures/upstream_stages.npz'
def test_upstream_stages():
 d=load()
 with np.load(P) as f:
  assert np.allclose(usage_penalties(d['sample_coordinates']),f['output_penalties'],rtol=2e-13,atol=2e-13)
  y=predict(f['input_fractions'],f['input_reactions'],d['endmembers'],d['reaction_ratios'])
  assert np.allclose(y,f['output_prediction'],rtol=2e-13,atol=2e-13)
  assert np.allclose(y-d['observations'][:12],f['output_residual'],rtol=1e-9,atol=2e-12)
