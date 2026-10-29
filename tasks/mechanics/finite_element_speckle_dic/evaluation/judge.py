"""Standalone displacement agreement evaluator, independent of the solver."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def score(prediction, reference):
    """RMSE in pixels; cosine NCC and dynamic-range NRMSE are diagnostics."""
    difference = prediction-reference
    rmse = float(np.sqrt(np.mean(difference**2)))
    denominator = float(np.linalg.norm(prediction)*np.linalg.norm(reference))
    if denominator == 0:
        ncc = 0.
    else:
        ncc = float(np.sum(prediction*reference)/denominator)
    dynamic_range = float(np.ptp(reference))
    if dynamic_range <= 0:
        raise ValueError('Reference displacement must have nonzero dynamic range')
    return dict(displacement_rmse_pixels=rmse, ncc=ncc, nrmse=rmse/dynamic_range)


def evaluate(prediction, asset_directory=None):
    assets = Path(asset_directory) if asset_directory is not None else Path(__file__).resolve().parent
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        settings = json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        boundary = settings['displacement_rmse_pixels_boundary']
        if not isinstance(boundary, (int, float)) or not np.isfinite(boundary) or boundary <= 0:
            raise ValueError('Invalid evaluator threshold')
        report['thresholds'] = {'displacement_rmse_pixels': {'comparison': '<=', 'boundary': boundary}}
        truth = assets/'ground_truth.npz'
        reference_path = truth if truth.exists() else assets/'baseline_reference.npz'
        with np.load(reference_path, allow_pickle=False) as data:
            reference = data['reconstruction'].copy()
        with np.load(prediction, allow_pickle=False) as data:
            value = data['reconstruction'].copy()
        for label, array in [('reference', reference), ('prediction', value)]:
            if array.shape != (1, 15, 2):
                raise ValueError(f'{label} shape must be (1, 15, 2)')
            if array.dtype not in (np.dtype('float32'), np.dtype('float64')):
                raise ValueError(f'{label} dtype must be float32 or float64')
            if not np.isfinite(array).all():
                raise ValueError(f'{label} contains nonfinite values')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            measured = score(value, reference)
        if not all(np.isfinite(number) for number in measured.values()):
            raise ValueError('Evaluation produced a nonfinite metric')
        report['metrics'] = measured
        report['passed'] = report['metrics']['displacement_rmse_pixels'] <= boundary
        report['status'] = 'pass' if report['passed'] else 'fail'
        return report, 0 if report['passed'] else 1
    except Exception as error:
        report['metrics'] = {}
        report['errors'].append(str(error))
        return report, 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    arguments = parser.parse_args()
    report, code = evaluate(arguments.prediction)
    try:
        target = Path(arguments.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as error:
        print(f'Cannot write judge report: {error}', file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
