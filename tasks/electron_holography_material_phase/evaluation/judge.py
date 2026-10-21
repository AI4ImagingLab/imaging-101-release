"""Independent standalone evaluator. No reference-solver imports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_SHAPE = (1, 512, 512)


def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Reference has zero or invalid norm')
    return float(np.linalg.norm((estimate - reference).ravel()) / denominator)


def observation_consistency(prediction: np.ndarray, raw_path: Path) -> float:
    """Compare both measured complex sidebands, using acquisition-time vacuum."""
    with np.load(raw_path, allow_pickle=False) as f:
        h, v = f['hologram'][0], f['vacuum'][0]
    # Fixed carrier convention and filter are declared in the output contract.
    yy = np.fft.fftfreq(512)[:, None]
    xx = np.fft.fftfreq(512)[None, :]
    radius = .5 * np.hypot(122, 58)
    window = (1 - np.tanh((np.hypot(yy, xx) - radius / 512) / (.025 * radius / 512))) / 2
    waves = []
    for image in (h, v):
        spectrum = np.fft.fft2(image.astype(np.float64))
        waves.append(np.fft.ifft2(np.roll(spectrum, (122, 454), axis=(0, 1)) * window))
    return relative_l2(prediction[0] * waves[1], waves[0])


def evaluate(prediction_path: Path, assets: Path = HERE) -> dict:
    result = {'status': 'error', 'passed': False, 'metrics': {}, 'thresholds': {}, 'errors': []}
    try:
        config = json.loads((assets / 'metrics.json').read_text(encoding='utf8'))
        for name in ('relative_complex_l2', 'sideband_relative_l2'):
            boundary = config[name + '_boundary']
            if not isinstance(boundary, (float, int)) or not np.isfinite(boundary) or boundary <= 0:
                raise ValueError(f'Invalid evaluator threshold: {name}')
            result['thresholds'][name] = {'comparison': '<=', 'boundary': boundary}
        with np.load(prediction_path, allow_pickle=False) as f:
            if 'reconstruction' not in f:
                raise ValueError('Missing required key reconstruction')
            prediction = f['reconstruction']
        if prediction.shape != EXPECTED_SHAPE:
            raise ValueError(f'Expected shape {EXPECTED_SHAPE}, got {prediction.shape}')
        if prediction.dtype not in (np.dtype('complex64'), np.dtype('complex128')):
            raise ValueError('Expected complex64 or complex128 reconstruction')
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite reconstruction values')
        reference_path = assets / 'ground_truth.npz'
        if not reference_path.exists():
            reference_path = assets / 'baseline_reference.npz'
        with np.load(reference_path, allow_pickle=False) as f:
            reference = f['reconstruction']
        if reference.shape != EXPECTED_SHAPE or reference.dtype.kind != 'c' or not np.isfinite(reference).all():
            raise ValueError('Invalid hidden reference')
        values = {'relative_complex_l2': relative_l2(prediction, reference),
                  'sideband_relative_l2': observation_consistency(prediction, assets.parent / 'data/raw_data.npz')}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Undefined evaluator metric')
        result['metrics'] = values
        result['passed'] = all(values[k] <= v['boundary'] for k, v in result['thresholds'].items())
        result['status'] = 'pass' if result['passed'] else 'fail'
    except Exception as exc:
        result['metrics'] = {}
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}', file=sys.stderr)
        return 2
    return {'pass': 0, 'fail': 1, 'error': 2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
