"""Audit complete independently executed runs, separate from staged fixtures."""
from pathlib import Path
import numpy as np

TASK = Path(__file__).resolve().parents[2]
REFERENCES = TASK / 'notebooks' / 'reference_outputs'


def test_complete_selected_adjust_against_independent_original():
    with np.load(TASK / 'output' / 'reconstruction.npz', allow_pickle=False) as fresh:
        with np.load(REFERENCES / 'upstream_ordered_corrected.npz', allow_pickle=False) as source:
            for key in ('reconstruction', 'spectra', 'dictionary_weights'):
                assert fresh[key].shape == source[key].shape
                assert fresh[key].dtype == source[key].dtype == np.float64
                np.testing.assert_allclose(fresh[key], source[key], rtol=2e-10, atol=2e-11)
    with np.load(REFERENCES / 'fresh_pipeline_measurements.npz', allow_pickle=False) as fresh:
        with np.load(REFERENCES / 'upstream_ordered_corrected.npz', allow_pickle=False) as source:
            np.testing.assert_allclose(fresh['history'], source['history'], rtol=2e-10, atol=2e-11)
        with np.load(REFERENCES / 'upstream_selfcalibration.npz', allow_pickle=False) as source:
            np.testing.assert_array_equal(fresh['quartz'], source['quartz'])


def test_complete_fresh_cjoint_against_independent_original():
    with np.load(REFERENCES / 'fresh_cleaned_measurements.npz', allow_pickle=False) as fresh:
        with np.load(REFERENCES / 'upstream_ordered_cjoint.npz', allow_pickle=False) as source:
            for key in ('maps', 'spectra'):
                actual, expected = fresh['cjoint_' + key], source[key]
                assert actual.shape == expected.shape
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
