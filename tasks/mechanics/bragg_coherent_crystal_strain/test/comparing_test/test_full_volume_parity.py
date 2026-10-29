"""Complete real-volume trajectory parity against independent corrected oracle."""
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_observations, prepare_observations
from src.solvers import ArrayBackend, reconstruct_one

TASK = Path(__file__).resolve().parents[2]


def test_complete_1020_step_real_p1_oracle():
    pytest.importorskip("cupy")
    backend = ArrayBackend("cupy")
    try:
        if backend.xp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("Full research trajectory requires a CUDA device")
    except backend.xp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable; full trajectory remains unverified")
    counts, _ = load_observations(TASK/"data/raw_data.npz")
    amplitude, _ = prepare_observations(counts[0])
    with np.load(TASK/"test/fixtures/full_volume_oracle.npz", allow_pickle=False) as archive:
        oracle = {key: archive[key] for key in archive.files}
    final_support = []

    def capture(step, image, support, kernel, error, active_backend):
        if step == 1020:
            final_support.append(active_backend.numpy(support))

    obj, kernel, errors = reconstruct_one(amplitude, device="cupy",
        initial=oracle["input_initial"], stage_callback=capture)
    np.testing.assert_allclose(obj, oracle["reconstruction"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(kernel, oracle["coherence"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(errors, oracle["errors"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(final_support[0], oracle["support"])
