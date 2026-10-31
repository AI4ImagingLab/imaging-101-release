from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data
from src.solvers import prepare_inverse,Objective,gauss_newton_direction

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('method',['smooth','tikhonov'])
def test_other_prior_complete_GN_with_original_captured_J(method):
    context=prepare_inverse(load_data(ROOT/'data/raw_data.npz'))
    objective=Objective(context,method)
    prediction=context.model.forward(context.initial_modulus)
    with np.load(ROOT/'test/fixtures/upstream_other_prior_updates.npz',allow_pickle=False) as fixture:
        direction=gauss_newton_direction(context.initial_modulus,prediction,fixture[method+'_J'],objective)
        reference=fixture[method+'_direction']
        # Identical captured J isolates prior/weighted algebra from FD cancellation.
        assert np.linalg.norm(direction-reference)/np.linalg.norm(reference)<1e-8
