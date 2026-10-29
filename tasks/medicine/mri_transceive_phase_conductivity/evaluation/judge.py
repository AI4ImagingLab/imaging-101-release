"""Standalone evaluator; never imports or executes submitted solver code."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (1, 96, 96, 80)


def evaluate(prediction):
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        config = json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))
        for key in ['ncc_boundary', 'nrmse_boundary']:
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError('Undefined or nonfinite metric boundary')
        if not -1 <= config['ncc_boundary'] <= 1 or config['nrmse_boundary'] < 0:
            raise ValueError('Metric boundary is out of range')
        with np.load(HERE / 'baseline_reference.npz', allow_pickle=False) as archive:
            reference = archive['reconstruction']
        with np.load(HERE.parent / 'data/raw_data.npz', allow_pickle=False) as archive:
            mask = archive['reconstruction_support']
        if reference.shape != SHAPE or reference.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.all(np.isfinite(reference)):
            raise ValueError('Invalid hidden reference shape/dtype/finiteness')
        if mask.shape != SHAPE or mask.dtype != np.bool_ or not mask.any():
            raise ValueError('Invalid public fixed-support asset')
        y = reference[mask].astype(np.float64)
        if np.ptp(y) <= 0 or np.linalg.norm(y) == 0:
            raise ValueError('Degenerate hidden reference; metric undefined')
    except Exception as exc:
        report['errors'] = ['Evaluator error: ' + str(exc)]
        return report
    report['thresholds'] = {'ncc': {'comparison': '>=', 'boundary': config['ncc_boundary']},
                            'nrmse': {'comparison': '<=', 'boundary': config['nrmse_boundary']}}
    try:
        path = Path(prediction)
        if not path.is_file():
            raise ValueError('Prediction file does not exist')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'reconstruction'}:
                raise ValueError('Prediction must contain exactly key reconstruction')
            estimate = archive['reconstruction']
        if estimate.shape != SHAPE:
            raise ValueError('Incorrect prediction shape; no reshape or crop permitted')
        if estimate.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Prediction dtype must be real float32 or float64')
        if not np.all(np.isfinite(estimate)):
            raise ValueError('Prediction contains NaN or Infinity')
        if np.any(estimate[~mask] != 0):
            raise ValueError('Prediction outside declared stencil support must be exactly zero')
    except Exception as exc:
        report['errors'] = ['Invalid prediction: ' + str(exc)]
        return report
    x = estimate[mask].astype(np.float64)
    norm = np.linalg.norm(x)
    ncc = float(np.dot(x, y) / norm / np.linalg.norm(y)) if norm else 0.0
    nrmse = float(np.sqrt(np.mean((x-y)**2)) / np.ptp(y))
    if not np.isfinite(ncc) or not np.isfinite(nrmse):
        report['errors'] = ['Invalid prediction: metrics overflowed numeric range']
        return report
    passed = ncc >= config['ncc_boundary'] and nrmse <= config['nrmse_boundary']
    report.update(status='pass' if passed else 'fail', passed=bool(passed), metrics={'ncc': ncc, 'nrmse': nrmse})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    report = evaluate(args.prediction)
    try:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as exc:
        print('Could not write report: ' + str(exc), file=sys.stderr)
        return 2
    return 0 if report['status'] == 'pass' else 1 if report['status'] == 'fail' else 2


if __name__ == '__main__':
    raise SystemExit(main())
