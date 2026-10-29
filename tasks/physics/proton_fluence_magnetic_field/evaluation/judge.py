"""Independent standalone evaluator. Never imports solver or executes submissions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (1, 2, 200, 200)

def load_array(path, role):
    if not Path(path).is_file():
        raise ValueError(f'{role}: file does not exist: {path}')
    with np.load(path, allow_pickle=False) as data:
        if 'reconstruction' not in data.files:
            raise ValueError(f'{role}: required key reconstruction is missing')
        value = data['reconstruction']
        if value.shape != SHAPE:
            raise ValueError(f'{role}: expected shape {SHAPE}, got {value.shape}')
        if value.dtype.kind != 'f' or value.dtype.itemsize not in (4, 8):
            raise ValueError(f'{role}: expected real float32 or float64')
        if not np.isfinite(value).all():
            raise ValueError(f'{role}: values must be finite')
        return value.astype(np.float64)

def component_metrics(estimate, reference):
    result = {}
    for i, name in enumerate(['Bx', 'By']):
        a, b = estimate[0, i].ravel(), reference[0, i].ravel()
        scale = np.ptp(b)
        norm = np.linalg.norm(b)
        if not scale > 0 or not norm > 0:
            raise ValueError('Evaluator reference is degenerate')
        anorm = np.linalg.norm(a)
        result[name + '_ncc'] = float(np.dot(a, b) / (anorm * norm)) if anorm > 0 else 0.0
        result[name + '_nrmse'] = float(np.sqrt(np.mean((a - b) ** 2)) / scale)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Metric overflow or undefined value')
    return result

def evaluate(prediction):
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        config = json.loads((HERE / 'metrics.json').read_text(encoding='utf-8'))
        actual = HERE / 'ground_truth.npz'
        reference_path = actual if actual.exists() else HERE / 'baseline_reference.npz'
        reference = load_array(reference_path, 'Evaluator reference')
        thresholds = {}
        for name in ['Bx', 'By']:
            for metric, direction in [('ncc', 'greater_equal'), ('nrmse', 'less_equal')]:
                key = name + '_' + metric
                value = config[key + '_boundary']
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f'Invalid evaluator boundary {key}')
                thresholds[key] = dict(comparison=direction, boundary=value)
        component_metrics(reference, reference)
    except Exception as error:
        report['errors'] = [f'Evaluator error: {error}']
        return report, 2
    try:
        estimate = load_array(prediction, 'Prediction')
        values = component_metrics(estimate, reference)
    except Exception as error:
        report['errors'] = [str(error)]
        return report, 2
    passed = all(values[k] >= t['boundary'] if t['comparison'] == 'greater_equal'
                 else values[k] <= t['boundary'] for k, t in thresholds.items())
    report.update(status='pass' if passed else 'fail', passed=bool(passed),
                  metrics=values, thresholds=thresholds)
    return report, 0 if passed else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report, code = evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as error:
        print(f'Cannot write evaluation report: {error}', file=sys.stderr)
        return 2
    return code

if __name__ == '__main__':
    sys.exit(main())
