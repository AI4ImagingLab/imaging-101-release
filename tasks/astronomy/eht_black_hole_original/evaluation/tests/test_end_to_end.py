"""Run the cleaned reconstruction and score its declared output."""

import numpy as np

from evaluation.judge import evaluate
from src.main import run_reconstruction


def test_fresh_reconstruction_passes_judge(tmp_path):
    image = run_reconstruction()
    assert image.shape == (64, 64)
    assert image.dtype == np.float64
    np.testing.assert_allclose(image.sum(), 0.6, rtol=1e-12)

    prediction = tmp_path / "reconstruction.npz"
    np.savez_compressed(prediction, reconstruction=image[None, ...])
    report = evaluate(prediction)
    assert report["status"] == "pass", report
