"""Standalone evaluator. Does not import solver code or execute predictions."""
import argparse
import json
from pathlib import Path
import sys
from zipfile import BadZipFile
import numpy as np

SHAPE = (1, 65, 257)


def load_array(path):
    if not Path(path).is_file():
        raise ValueError(f'File unavailable: {path}')
    with np.load(path, allow_pickle=False) as archive:
        if 'reconstruction' not in archive:
            raise ValueError(f'Missing reconstruction key: {path}')
        array = archive['reconstruction']
    if array.shape != SHAPE:
        raise ValueError(f'reconstruction shape must be {SHAPE}, got {array.shape}')
    if array.dtype != np.float64:
        raise ValueError(f'reconstruction dtype must be float64, got {array.dtype}')
    if not np.isfinite(array).all():
        raise ValueError('reconstruction must contain only finite numbers')
    return array


def compute_metrics(estimate, reference):
    """Cosine NCC (zero estimate -> 0); reference-range-normalized RMSE."""
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if (estimate.shape != reference.shape or estimate.size == 0
            or np.iscomplexobj(estimate) or np.iscomplexobj(reference)
            or not np.isfinite(estimate).all() or not np.isfinite(reference).all()):
        raise ValueError('Metrics require matching finite real arrays')
    dynamic_range = float(np.max(reference)-np.min(reference))
    ref_scale = float(np.max(np.abs(reference)))
    if dynamic_range <= 0 or not np.isfinite(dynamic_range) or ref_scale == 0:
        raise ValueError('Reference has undefined dynamic-range NRMSE or NCC')
    ref = reference.ravel()/ref_scale
    est_scale = float(np.max(np.abs(estimate)))
    if est_scale == 0:
        ncc = 0.0
    else:
        est = estimate.ravel()/est_scale
        ncc = float(np.dot(est, ref)/(np.linalg.norm(est)*np.linalg.norm(ref)))
    with np.errstate(over='raise', invalid='raise'):
        difference = estimate-reference
        scale = float(np.max(np.abs(difference)))
        nrmse = 0.0 if scale == 0 else float((scale/dynamic_range)*np.sqrt(np.mean((difference/scale)**2)))
    if not np.isfinite([ncc, nrmse]).all():
        raise ValueError('Metrics are not finite')
    return {'ncc': ncc, 'nrmse': nrmse}


def evaluate(prediction, assets=None):
    assets = Path(assets) if assets is not None else Path(__file__).resolve().parent
    report = {'status': 'error', 'passed': False, 'metrics': {}, 'thresholds': {}, 'errors': []}
    try:
        config = json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        ncc = config['ncc_boundary']
        nrmse = config['nrmse_boundary']
        if (isinstance(ncc, bool) or isinstance(nrmse, bool)
                or not np.isfinite([ncc, nrmse]).all() or not -1 <= ncc <= 1 or nrmse <= 0):
            raise ValueError('Undefined or invalid metric thresholds')
        report['thresholds'] = {'ncc': {'comparison': '>=', 'boundary': ncc},
                                 'nrmse': {'comparison': '<=', 'boundary': nrmse}}
        truth = assets/'ground_truth.npz'
        reference = load_array(truth if truth.exists() else assets/'baseline_reference.npz')
        estimate = load_array(prediction)
        measured = compute_metrics(estimate, reference)
        passed = bool(measured['ncc'] >= ncc and measured['nrmse'] <= nrmse)
        report.update(status='pass' if passed else 'fail', passed=passed, metrics=measured)
        return report, 0 if passed else 1
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError, OverflowError, BadZipFile, EOFError) as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        return report, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report, code = evaluate(args.prediction)
    try:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except (OSError, ValueError, TypeError) as error:
        print(f'Unable to write report: {error}', file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    sys.exit(main())
