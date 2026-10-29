from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct,compare_formulations

ROOT=Path(__file__).resolve().parents[2]

def test_actual_paper_comparisons_against_separate_author_runs():
    result=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    fresh=compare_formulations(result,include_operator=True)
    original=np.load(ROOT/'notebooks/reference_outputs/original_formulations.npz',allow_pickle=False)
    routes={'young_dupre':'YoungDupre','projection':'ProjectionYoungDupre','lami':'Lami',
            'inverse_lami':'InvLami','log_lami':'LogLami','variational_operator':'Variational'}
    for local,source in routes.items():
        for key in ['tensions','pressures']:
            np.testing.assert_allclose(fresh[local][key],original[source+'_'+key],rtol=2e-9,atol=2e-10)
    for local,source in [('laplace','Laplace'),('laplace_weighted_curvature','WeightedLaplace')]:
        np.testing.assert_allclose(fresh[local]['pressures'],original[source+'_pressures'],rtol=2e-9,atol=2e-10)
    gram=np.load(ROOT/'notebooks/reference_outputs/original_gram.npz',allow_pickle=False)
    for key in ['tensions','pressures']:
        np.testing.assert_allclose(fresh['variational_gram'][key],gram[key],rtol=2e-9,atol=2e-10)
