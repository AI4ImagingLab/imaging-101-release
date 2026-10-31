"""Fixed actual upstream draws isolate arithmetic/stopping from random streams."""
from pathlib import Path
import numpy as np
from src.solvers import fit_phase_plane


class Replay:
    def __init__(self, values):
        self.values, self.position = values, 0

    def choice(self, population, size, replace):
        assert size == 3 and replace is False
        result = self.values[self.position].copy()
        self.position += 1
        return result


def test_upstream_actual_draws_and_consensus_refit():
    root = Path(__file__).resolve().parents[1] / 'fixtures'
    with np.load(root/'upstream_fixed_triplets.npz', allow_pickle=False) as a, np.load(root/'upstream_ransac_statistics.npz', allow_pickle=False) as f:
        for case in range(4):
            x, y, phase = [f[f'input_{k}_{case}'] for k in ['x', 'y', 'phase']]
            design = np.column_stack([x, y, np.ones(len(x))])
            for repeat in range(8):
                key = f'{case}_{repeat}'
                triples = a[f'input_triples_{key}']
                coefficients, indices, diagnostics = fit_phase_plane(x, y, phase, Replay(triples))
                np.testing.assert_allclose(coefficients, a[f'output_coefficients_{key}'], atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(indices, a[f'output_indices_{key}'])
                draws = int(a[f'output_draws_{key}'][0])
                nonsingular = 0
                for triple in triples[:draws]:
                    sample = design[triple]
                    try:
                        np.linalg.solve(sample.T @ sample, sample.T @ phase[triple])
                        nonsingular += 1
                    except np.linalg.LinAlgError:
                        pass
                assert diagnostics['draws'] == draws
                assert diagnostics['trials'] == nonsingular - int(len(indices) == len(x))
