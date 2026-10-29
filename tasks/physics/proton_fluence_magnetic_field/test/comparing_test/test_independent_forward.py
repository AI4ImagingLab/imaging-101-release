from pathlib import Path
import numpy as np
from src.physics_model import ray_forward_fluence
from src.preprocessing import load_observations

ROOT=Path(__file__).resolve().parents[2]

def test_independent_world_basis_ray_forward_both_signs():
    data=load_observations()
    with np.load(ROOT/'evaluation/ground_truth.npz',allow_pickle=False) as f:truth=f['reconstruction'][0]
    with np.load(ROOT/'test/fixtures/independent_rays.npz',allow_pickle=False) as f:
        for direction,name in [(-1,'source_into_screen'),(1,'opposite_beam_sign')]:
            result=ray_forward_fluence(data['x'],data['y'],truth,np.mean(data['fluence']),subpixels=int(f['param_subpixels']),beam_direction=direction)
            np.testing.assert_array_equal(result,f['output_'+name])
            np.testing.assert_allclose(result.sum(),data['fluence'].sum(),rtol=3e-15)
