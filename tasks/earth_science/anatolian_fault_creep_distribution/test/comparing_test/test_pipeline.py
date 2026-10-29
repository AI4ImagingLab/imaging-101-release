from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_observations
from src.solvers import assemble_problem,solve_system

ROOT=Path(__file__).resolve().parents[2]


def test_all_three_original_inverse_comparisons_and_kkt():
    data=load_observations(ROOT/'data/raw_data.npz')
    system=assemble_problem(data)
    # Independently measured original float32-serialization envelope3.00022e-7,
    # factor1.5; canonical final output additionally has stricter per-group gates.
    absolute=4.500325418987927e-7
    with np.load(ROOT/'notebooks/reference_outputs/original_comparisons.npz',allow_pickle=False) as source:
        for method in ['gls','nnls','tied_nnls']:
            m,details=solve_system(*system,method=method)
            assert_allclose(m,source['original_'+method],atol=absolute,rtol=0)
            if method!='gls':
                assert details['projected_gradient_inf']<1e-7
                assert np.min(details['reduced_model']-details['prior'])>=-1e-12
            if method=='tied_nnls':
                assert np.ptp(m[1292:1297])==0.
                assert details['weighted_observation_rms']<1.2399866256976022
