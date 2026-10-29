"""Independent strict opacity evaluator; imports no reference solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np

ASSETS = Path(__file__).resolve().parent
SHAPE = (1, 440, 680)


def read_array(path):
    if not Path(path).is_file():
        raise ValueError('missing NPZ: '+str(path))
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'reconstruction'}:
            raise ValueError('exact reconstruction key required')
        x = z['reconstruction']
    if x.shape != SHAPE or x.dtype != np.float64 or not np.isfinite(x).all():
        raise ValueError('expected finite real float64 reconstruction with shape '+str(SHAPE))
    return x


def image_metrics(x, reference):
    span = float(np.ptp(reference)); norm = np.linalg.norm(reference.ravel())
    if not np.isfinite(span) or span <= 0 or not np.isfinite(norm) or norm == 0:
        raise ValueError('evaluator reference is degenerate')
    estimate_norm = np.linalg.norm(x.ravel())
    ncc = float(np.dot(x.ravel(), reference.ravel())/(estimate_norm*norm)) if estimate_norm else 0.
    nrmse = float(np.sqrt(np.mean((x-reference)**2))/span)
    return {'ncc': ncc, 'nrmse': nrmse}


def evaluate(prediction):
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[])
    try:
        configuration = json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'))
        directions = dict(ncc='>=', nrmse='<=', known_pixel_max_error='<=')
        expected = {name+'_boundary' for name in directions}
        if {k for k in configuration if k.endswith('_boundary')} != expected:
            raise ValueError('missing or unexpected metric boundary')
        for name, comparison in directions.items():
            boundary = configuration[name+'_boundary']
            if isinstance(boundary, bool) or not isinstance(boundary, (int,float)) or not np.isfinite(boundary):
                raise ValueError('invalid metric boundary: '+name)
            if comparison == '<=' and boundary < 0:
                raise ValueError('negative error boundary: '+name)
            if name == 'ncc' and not -1 <= boundary <= 1:
                raise ValueError('NCC boundary outside [-1,1]')
            report['thresholds'][name] = dict(comparison=comparison, boundary=boundary)
        target = ASSETS/'ground_truth.npz'
        if not target.exists(): target = ASSETS/'baseline_reference.npz'
        reference = read_array(target)
        x = read_array(prediction)
        if np.any((x < -1e-10)|(x > 1+1e-10)):
            raise ValueError('opacity outside physical [0,1] interval')
        with np.load(ASSETS.parent/'data/raw_data.npz', allow_pickle=False) as raw:
            trimap = raw['trimap']
        if trimap.shape != SHAPE or trimap.dtype != np.float64 or not np.isfinite(trimap).all():
            raise ValueError('evaluator public constraint asset invalid')
        known = (trimap == 0)|(trimap == 1)
        if not np.any(known): raise ValueError('no legitimate known opacity constraints')
        measured = image_metrics(x, reference)
        measured['known_pixel_max_error'] = float(abs(x[known]-trimap[known]).max())
        if not all(np.isfinite(value) for value in measured.values()):
            raise ValueError('nonfinite computed metric')
        passed = all(measured[name] >= item['boundary'] if item['comparison']=='>=' else measured[name] <= item['boundary'] for name,item in report['thresholds'].items())
        report.update(status='pass' if passed else 'fail', passed=bool(passed), metrics=measured)
        return report, 0 if passed else 1
    except Exception as error:
        report['errors'] = [str(error)]; return report, 2


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prediction',required=True); p.add_argument('--report',required=True); args = p.parse_args()
    report, code = evaluate(args.prediction)
    try:
        destination = Path(args.report); destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False),encoding='utf8')
    except Exception as error:
        print('report write failed: '+str(error),file=sys.stderr); return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
