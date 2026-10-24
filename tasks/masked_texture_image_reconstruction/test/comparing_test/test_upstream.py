from pathlib import Path
import numpy as np
import pytest
from src import physics_model as model
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('name,input_key',[('gradient','probe'),('gradient_adjoint','probe_vector'),('symgrad','probe_vector'),('symgrad_adjoint','probe_tensor'),('symgrad2','probe_tensor'),('symgrad2_adjoint','probe_third')])
def test_frozen_original_operators(name,input_key):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(getattr(model,name)(z[input_key]),z[name],rtol=0,atol=1e-12)
@pytest.mark.parametrize('order',[1,2,3])
def test_original_updates(order):
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        result=reconstruct(z['tiny_observations'],z['tiny_mask'],order=order,niter=30,capture_steps=(1,2,5,10,30))
        for step,fields in result['snapshots'].items():
            for name,value in fields.items():np.testing.assert_allclose(value,z[f'order{order}_step{step}_{name}'],rtol=0,atol=1e-11)
        np.testing.assert_allclose(result['objective'],z[f'tiny_objective{order}'],rtol=0,atol=1e-11)

