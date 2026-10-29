from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct
T=Path(__file__).resolve().parents[2]
def test_original_full_pipeline_all_comparison_methods():
    y,A,mask=load_observations(T/'data/raw_data.npz')
    with np.load(T/'notebooks/reference_outputs/original_methods.npz') as z:
        for method in ['l2','l1','sbl','rpca']:
            a=reconstruct(y,A,mask,method)
            np.testing.assert_allclose(a,z[method][0],rtol=1e-9,atol=1e-10)
