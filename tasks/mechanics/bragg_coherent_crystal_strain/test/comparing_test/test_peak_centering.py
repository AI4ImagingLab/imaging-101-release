"""Frozen unmodified upstream peak rolls on both exact experimental scans."""
from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, measured_amplitude, center_diffraction_peak
from src.physics_model import observation_intensity, raw_observation_intensity

TASK = Path(__file__).resolve().parents[2]


def test_centering_matches_original_on_both_full_observations():
    counts, _ = load_observations(TASK / 'data/raw_data.npz')
    with np.load(TASK / 'test/fixtures/upstream_center_max.npz', allow_pickle=False) as reference:
        for position, name in enumerate(('P1', 'P2')):
            amplitude = measured_amplitude(counts[position])
            centered, shift = center_diffraction_peak(amplitude)
            np.testing.assert_array_equal(shift, reference[name + '_shift'])
            np.testing.assert_array_equal(centered, reference[name + '_amplitude'])
            np.testing.assert_array_equal(np.roll(centered, -shift, axis=(0, 1, 2)), amplitude)
            assert np.argmax(centered) == np.ravel_multi_index(tuple(np.array(centered.shape)//2), centered.shape)
            centered_counts = np.roll(counts[position], shift, axis=(0, 1, 2))
            assert centered_counts.sum(dtype=np.int64) == counts[position].sum(dtype=np.int64)


def test_raw_detector_operator_undoes_fixed_peak_roll():
    # A unit object voxel has constant diffraction intensity; a nontrivial pair
    # additionally checks axis/sign conventions without using cleaned fixtures.
    obj = np.zeros((7, 8, 9), dtype=np.complex128)
    obj[3, 4, 4] = 1
    obj[2, 5, 6] = 0.3 + 0.2j
    kernel = np.ones((2, 3, 2), dtype=np.float64) / 12
    shift = np.array((1, -2, 3))
    centered = observation_intensity(obj, kernel)
    expected = centered[np.ix_((np.arange(7)+1)%7, (np.arange(8)-2)%8, (np.arange(9)+3)%9)]
    np.testing.assert_array_equal(raw_observation_intensity(obj, kernel, shift), expected)
