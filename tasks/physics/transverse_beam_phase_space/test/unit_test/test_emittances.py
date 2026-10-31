from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.physics_model import emittances
ROOT=Path(__file__).resolve().parents[2]

def test_geometric_emittances_independent_known_spd_and_original_diagnostic():
    result=emittances(np.diag([4.,9.,25.,36.]))
    assert_allclose([result[k] for k in ['eps_x','eps_y','eps_1','eps_2']],[6,30,30,6],rtol=1e-12,atol=1e-12)
    z=np.load(ROOT/'notebooks/reference_outputs/source_eval_metrics.npz',allow_pickle=False)
    result=emittances(z['physical_covariance'])
    for key,value in result.items():assert_allclose(value,z[key],rtol=1e-9,atol=1e-10)
