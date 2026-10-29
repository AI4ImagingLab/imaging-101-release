"""Independent, strict NPZ stereo evaluator. No submitted code is imported."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (1, 496, 718)
KEYS = {'disparity', 'reconstruction'}

def evaluate(prediction):
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        config = json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        expected = {'disparity_ncc':'>=', 'disparity_nrmse':'<=', 'depth_ncc':'>=',
                    'depth_nrmse':'<=', 'disparity_bad1_fraction':'<=', 'geometry_relative_error':'<='}
        thresholds = config['thresholds']
        if set(thresholds) != set(expected):
            raise ValueError('Missing or unexpected metric definitions')
        for name, direction in expected.items():
            entry = thresholds[name]
            value = entry['boundary']
            if entry['comparison'] != direction or isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value):
                raise ValueError('Invalid metric boundary '+name)
            if name.endswith('ncc') and not -1 <= value <= 1:
                raise ValueError('NCC boundary outside [-1,1]')
            if not name.endswith('ncc') and value < 0:
                raise ValueError('Negative error boundary')
            if config[name+'_boundary'] != value:
                raise ValueError('Inconsistent harness boundary')
        with np.load(HERE/'ground_truth.npz', allow_pickle=False) as data:
            mask = data['valid_mask']
            reference = {key:data[key] for key in KEYS}
        if mask.shape != SHAPE or mask.dtype != np.bool_ or not mask.any():
            raise ValueError('Invalid reference availability mask')
        for key, array in reference.items():
            if array.shape != SHAPE or array.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError('Invalid hidden shape/dtype')
            values = array[mask].astype(np.float64)
            if not np.isfinite(values).all() or np.ptp(values) <= 0 or np.linalg.norm(values) == 0:
                raise ValueError('Nonfinite or degenerate measured reference')
    except Exception as exc:
        report['errors'] = ['Evaluator error: '+str(exc)]
        return report
    report['thresholds'] = thresholds
    try:
        with np.load(Path(prediction), allow_pickle=False) as data:
            if set(data.files) != KEYS:
                raise ValueError('Required keys are exactly disparity and reconstruction')
            estimate = {key:data[key] for key in KEYS}
        for key, array in estimate.items():
            if array.shape != SHAPE:
                raise ValueError('Incorrect shape for '+key)
            if array.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError('Expected real float32/float64 for '+key)
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite prediction '+key)
        denominator = estimate['disparity'].astype(np.float64)+53.271
        if np.any(denominator <= 0) or np.any(estimate['reconstruction'] <= 0):
            raise ValueError('Nonpositive physical depth or stereo denominator')
        values = {}
        for key, prefix in [('disparity','disparity'),('reconstruction','depth')]:
            x = estimate[key][mask].astype(np.float64)
            y = reference[key][mask].astype(np.float64)
            norm = np.linalg.norm(x)
            values[prefix+'_ncc'] = float(np.dot(x,y)/norm/np.linalg.norm(y)) if norm else 0.
            values[prefix+'_nrmse'] = float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))
        values['disparity_bad1_fraction'] = float(np.mean(np.abs(estimate['disparity'][mask].astype(float)-reference['disparity'][mask]) > 1.))
        physical_depth = 1038.018*176.252/denominator
        values['geometry_relative_error'] = float(np.max(np.abs(estimate['reconstruction'].astype(float)-physical_depth)/physical_depth))
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('Metric arithmetic overflow')
    except Exception as exc:
        report['errors'] = ['Invalid prediction: '+str(exc)]
        return report
    passed = all(values[name] >= entry['boundary'] if entry['comparison']=='>=' else values[name] <= entry['boundary'] for name,entry in thresholds.items())
    report.update(status='pass' if passed else 'fail', passed=bool(passed), metrics=values)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    report = evaluate(args.prediction)
    try:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc), file=sys.stderr)
        return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2

if __name__ == '__main__':
    raise SystemExit(main())
