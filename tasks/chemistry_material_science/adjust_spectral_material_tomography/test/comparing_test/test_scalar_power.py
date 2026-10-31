"""Actual original scalar powers, frozen before cleaned primitive validation."""
from pathlib import Path
import numpy as np
from src.linear_algebra import SourceBlas
from src.solvers import (_cubic_step, _scalar_power, frobenius_norm,
                         project_dictionary_weights)


FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/upstream_scalar_power.npz'


def test_independent_original_power_on_actual_norms_and_boundaries():
    with np.load(FIXTURE, allow_pickle=False) as original, SourceBlas() as backend:
        values = np.array([backend.power(base, exponent)
                           for base, exponent in zip(original['base'], original['exponent'])])
        np.testing.assert_array_equal(values.view(np.uint64), original['power'].view(np.uint64))
        actual_norm = float(original['base'][1])
        source_square = float(original['power'][1])
        assert _scalar_power(actual_norm, 2.) == source_square
        assert source_square != actual_norm * actual_norm
    assert backend.process.returncode == 0


def test_original_cubic_steps_on_all_actual_101_to_164_operands():
    with np.load(FIXTURE, allow_pickle=False) as original, SourceBlas():
        actual = np.array([_cubic_step(p[0, 1], p[0, 2], p[1, 0], p[1, 1], p[1, 2])
                           for p in original['cubic_inputs']])
        np.testing.assert_array_equal(actual.view(np.uint64),
                                      original['cubic_expected'].view(np.uint64))
        np.testing.assert_array_equal(original['cubic_d1_power'].view(np.uint64),
                                      original['cubic_d1_multiply'].view(np.uint64))


def test_actual_original_checkpoint100_and300_norms_and_projections():
    fixture = FIXTURE.with_name('upstream_checkpoint100300.npz')
    with np.load(fixture, allow_pickle=False) as original:
        norms = np.array([frobenius_norm(original[key])
                          for key in ('A100', 'R100', 'A300', 'R300')])
        np.testing.assert_array_equal(norms.view(np.uint64),
                                      original['source_norms'].ravel().view(np.uint64))
        for iteration in (100, 300):
            actual = project_dictionary_weights(original[f'R{iteration}'])
            np.testing.assert_array_equal(actual.view(np.uint64),
                                          original[f'P{iteration}'].view(np.uint64))
