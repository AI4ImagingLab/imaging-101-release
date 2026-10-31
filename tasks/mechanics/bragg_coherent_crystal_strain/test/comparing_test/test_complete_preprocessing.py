"""Independent complete upstream prep on the exact two bound observations."""
from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, prepare_observations

TASK = Path(__file__).resolve().parents[2]


def test_complete_upstream_preprocessing_exact():
    counts, _ = load_observations(TASK/'data/raw_data.npz')
    original_sums = np.sum(counts, axis=(1,2,3), dtype=np.int64)
    with np.load(TASK/'test/fixtures/complete_preprocessing.npz',allow_pickle=False) as f:
        for position in range(2):
            amplitude, metadata = prepare_observations(counts[position])
            assert amplitude.shape == (126,256,256) and amplitude.dtype == np.float32
            np.testing.assert_array_equal(amplitude,f['amplitude'][0,position])
            np.testing.assert_array_equal(metadata['peak_shift'],f['peak_shifts'][position])
            np.testing.assert_allclose(metadata['intensity_threshold'],f['thresholds'][position],rtol=0,atol=1e-14)
            assert metadata['padding'] == ((2,3),(0,0),(0,0))
    np.testing.assert_array_equal(np.sum(counts,axis=(1,2,3),dtype=np.int64),original_sums)
