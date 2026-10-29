from pathlib import Path
import numpy as np,torch
from src.preprocessing import load_observations
from src.solvers import reconstruct
from src.physics_model import carve,project
T=Path(__file__).resolve().parents[2]

def test_native_carving_all_views_both_conditions():
    images,prior=load_observations(T/'data/raw_data.npz')
    with np.load(T/'evaluation/ground_truth.npz') as z:true=z['viewpoints'][0]
    with np.load(T/'test/fixtures/upstream_carving.npz') as source:
        for name,angles in [('noisy',prior),('true',true)]:
            result=carve(images,angles);np.testing.assert_array_equal(result,source['output_'+name+'_volume'])

def test_frozen_original_500_step_pipeline_and_first_update(tmp_path):
    images,prior=load_observations(T/'data/raw_data.npz')
    volume,views,history,info=reconstruct(images,prior,capture_stages=True)
    with np.load(T/'test/fixtures/upstream_stages.npz') as source:
        for k,v in info['stages'].items():np.testing.assert_allclose(v,source[k],rtol=0,atol=2e-6,err_msg=k)
    with np.load(T/'test/fixtures/upstream_full.npz') as source:
        np.testing.assert_array_equal(volume,source['output_reconstruction']);np.testing.assert_array_equal(views,source['output_views']);np.testing.assert_array_equal(history,source['output_losses']);np.testing.assert_array_equal(info['projections'],source['output_projections'])
