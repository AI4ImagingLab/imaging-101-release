"""Independent original initial-R feasibility trajectory, including active size 3."""
from pathlib import Path

import numpy as np

from src.solvers import project_dictionary_weights, project_simplex_columns


def test_original_initial_projection_every_half_step():
    path = Path(__file__).resolve().parents[1] / 'fixtures/upstream_initial_projection.npz'
    with np.load(path, allow_pickle=False) as source:
        current = source['initial'].copy()
        iterations = int(source['iterations'].item())
        assert iterations == 39
        assert np.any(np.count_nonzero(source['row_projections'], axis=1) == 3)
        for index in range(iterations):
            row = project_simplex_columns(current.T, equality=True).T
            np.testing.assert_array_equal(row, source['row_projections'][:, :, index])
            current = project_simplex_columns(row)
            np.testing.assert_array_equal(current, source['column_projections'][:, :, index])
        np.testing.assert_array_equal(current, source['result'])
        np.testing.assert_array_equal(project_dictionary_weights(source['initial']), source['result'])
