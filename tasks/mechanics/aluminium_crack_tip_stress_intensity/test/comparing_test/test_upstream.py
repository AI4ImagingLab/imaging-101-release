from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_pipeline_stages():
    stress,k,d=reconstruct(*load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for name,key,tol in [('points','coordinates',1e-13),('displacements','displacements',1e-13),('x','grid_x',1e-13),('y','grid_y',1e-13),('sampled_x','interp_x',1e-12),('sampled_y','interp_y',1e-12),('A','system',1e-14),('b','target',1e-12),('coefficients','coefficients',1e-7),('residual','residual',1e-12)]:
            np.testing.assert_allclose(d[name],z['output_'+key],atol=tol,rtol=1e-8,equal_nan=True)
        np.testing.assert_allclose(stress,z['output_stress'],atol=1e-6,rtol=1e-8)
        np.testing.assert_allclose(k,z['output_stress_intensity'],atol=1e-8,rtol=1e-8)
