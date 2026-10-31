"""Disk storage changes lifetimes, never start selection or fresh execution."""
from pathlib import Path
import numpy as np
from src import preprocessing, solvers


def test_selected_storage_preserves_ranking_and_calls_every_fresh_start(monkeypatch):
    shape = (5, 6, 7)
    amplitude = np.ones(shape, dtype=np.float32)
    metadata = {"peak_shift": np.zeros(3, dtype=int), "intensity_threshold": 2.,
                "padding": ((0, 0),)*3}
    monkeypatch.setattr(preprocessing, "prepare_observations", lambda _: (amplitude, metadata))
    calls, paths = [], []
    original_storage = solvers.StoredObjects

    def storage(items):
        paths.extend(items)
        return original_storage(items)

    def synthetic_stage(seed):
        rng = np.random.default_rng(seed)
        obj = rng.random(shape)*np.exp(.1j*rng.random(shape))
        return obj, np.ones((1, 1, 1)), np.array([float((seed*7) % 11)])

    def fresh_stage(_, seed, device):
        calls.append(seed)
        return synthetic_stage(seed)

    monkeypatch.setattr(solvers, "StoredObjects", storage)
    monkeypatch.setattr(solvers, "reconstruct_one", fresh_stage)
    result, kernels, histories = solvers.reconstruct(np.ones((2,)+shape), starts=12, device="cpu")
    assert calls == list(range(10000, 10012))+list(range(11000, 11012))
    for position in range(2):
        candidates = []
        for start in range(12):
            obj, kernel, error = synthetic_stage(10000+position*1000+start)
            candidates.append((error[-1], start, obj, kernel))
        selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:10]
        expected, coherence, alignment = solvers.aggregate_ensemble(
            [item[2] for item in selected], [item[3] for item in selected], amplitude)
        np.testing.assert_array_equal(result[0, position], expected)
        np.testing.assert_array_equal(kernels[0, position], coherence)
        assert histories[position]["selected"] == [item[1] for item in selected]
        assert histories[position]["alignment"] == alignment
    assert len(paths) == 20
    assert all(not Path(path).exists() for path in paths)
