"""Independent standalone evaluator for the fixed real SH coefficient contract."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np


def score(estimate, reference):
    """Relative L2 error over all coefficients; reference norm must be nonzero."""
    norm = np.linalg.norm(reference.ravel())
    if norm <= 0:
        raise ValueError('Reference has zero norm')
    return float(np.linalg.norm((estimate-reference).ravel()) / norm)


def evaluate(prediction, asset_dir=None):
    """Return JSON-compatible report and process exit status."""
    assets = Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        config = json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        boundary = config['relative_l2_boundary']
        if not np.isfinite(boundary) or boundary <= 0:
            raise ValueError('Invalid metric boundary')
        report['thresholds'] = {'relative_l2': {'comparison':'<=', 'boundary':boundary}}
        truth = assets/'ground_truth.npz'
        ref_path = truth if truth.exists() else assets/'baseline_reference.npz'
        with np.load(ref_path, allow_pickle=False) as data:
            ref = data['reconstruction']
        if ref.shape != (1,24,24,1,45) or ref.dtype != np.float64 or not np.isfinite(ref).all():
            raise ValueError('Invalid evaluator reference')
        with np.load(prediction, allow_pickle=False) as data:
            if 'reconstruction' not in data:
                raise ValueError('Missing required reconstruction key')
            estimate = data['reconstruction']
        if estimate.shape != ref.shape:
            raise ValueError('Expected shape (1,24,24,1,45)')
        if estimate.dtype != np.float64:
            raise ValueError('Expected real float64 dtype')
        if not np.isfinite(estimate).all():
            raise ValueError('Prediction contains NaN or Infinity')
        error = score(estimate, ref)
        report['metrics'] = {'relative_l2':error}
        report['passed'] = error <= boundary
        report['status'] = 'pass' if report['passed'] else 'fail'
        return report, 0 if report['passed'] else 1
    except Exception as exc:
        report['errors'].append(str(exc))
        return report, 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report, code = evaluate(args.prediction)
    try:
        target = Path(args.report); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}', file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    sys.exit(main())
