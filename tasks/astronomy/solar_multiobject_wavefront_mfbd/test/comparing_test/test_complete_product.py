"""Full product, independently captured parity and honest review-path checks."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_fresh_full_product_passes_independent_evaluator():
    from evaluation.judge import evaluate
    report, code = evaluate(ROOT/'output/reconstruction.npz')
    assert code == 0, report
    assert len(report['metrics']) == 9


def test_full_loss_history_matches_independent_original():
    with np.load(ROOT/'notebooks/reference_outputs/upstream_measurements.npz', allow_pickle=False) as upstream:
        original = upstream['loss_history'][0]
    with np.load(ROOT/'notebooks/reference_outputs/fresh_measurements.npz', allow_pickle=False) as fresh:
        actual = fresh['loss_history'][0]
        assert fresh['runtime_seconds'][0] > 0
    assert original.shape == actual.shape == (250,)
    np.testing.assert_allclose(actual, original, rtol=2e-6, atol=3e-5)


def test_notebook_has_active_complete_path_and_six_sections():
    notebook = json.loads((ROOT/'notebooks/solar_multiobject_wavefront_mfbd.ipynb').read_text(encoding='utf-8'))
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
    assert [line for line in markdown.splitlines() if line.startswith('## ')] == [
        '## 1. Problem', '## 2. Data', '## 3. Physics', '## 4. Method', '## 5. Results', '## 6. Conclusion']
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['execution_count'] is not None for cell in code)
    assert not any(output.get('output_type') == 'error' for cell in code for output in cell['outputs'])
    assert sum('image/png' in output.get('data', {}) for cell in code for output in cell['outputs']) == 5
    assert 'fresh = reconstruct(observations, mixing, device=device, progress=progress)' in '\n'.join(''.join(cell['source']) for cell in code)
