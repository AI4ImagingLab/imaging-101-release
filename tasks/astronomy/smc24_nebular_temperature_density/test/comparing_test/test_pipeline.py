from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,observation_ratios
from src.physics_model import load_ions
from src.solvers import crossed_solution

def test_deredden_and_inverse():
    root=Path(__file__).parents[2]
    raw,corrected=load_observations(root/'data/raw_data.npz')
    with np.load(root/'notebooks/reference_outputs/upstream_original.npz',allow_pickle=False) as a:
        for key,value in corrected.items():
            np.testing.assert_allclose(value,a['corrected_'+key][0],rtol=1e-10,atol=0,err_msg=key)
        ions=load_ions(root/'data/raw_data.npz');r=observation_ratios(corrected)
        actual=np.array([crossed_solution(ions,t,n,r[t],r[n]) for t in ['NII','OIII'] for n in ['OII','SII']])
        np.testing.assert_allclose(actual,a['output_conditions'],rtol=1e-8,atol=0)
