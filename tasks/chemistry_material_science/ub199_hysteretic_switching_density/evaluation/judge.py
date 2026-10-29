"""Independent standalone evaluator; imports no solver code."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ASSETS=Path(__file__).resolve().parent
KEYS=('reconstruction','standard_error')
SHAPE=(1,11310)


def load_arrays(path):
    if not Path(path).is_file():
        raise ValueError(f'NPZ file does not exist: {path}')
    with np.load(path,allow_pickle=False) as source:
        missing=set(KEYS)-set(source.files)
        if missing:
            raise ValueError(f'Missing required arrays: {sorted(missing)}')
        arrays={k:source[k] for k in KEYS}
    for key,a in arrays.items():
        if a.shape!=SHAPE:
            raise ValueError(f'{key}: shape must be {SHAPE}, received {a.shape}')
        if a.dtype!=np.dtype('float64'):
            raise ValueError(f'{key}: dtype must be real float64, received {a.dtype}')
        if not np.isfinite(a).all():
            raise ValueError(f'{key}: nonfinite values')
    if (arrays['standard_error']<0).any():
        raise ValueError('standard_error must be nonnegative')
    return arrays


def evaluate(prediction, assets=ASSETS):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        for key in KEYS:
            for metric,comparison in [('ncc','>='),('nrmse','<=')]:
                name=f'{key}_{metric}'
                boundary=config[f'{name}_boundary']
                if not isinstance(boundary,(int,float)) or not math.isfinite(boundary):
                    raise ValueError(f'Invalid evaluator boundary: {name}')
                if (metric=='ncc' and not 0<boundary<=1) or (metric=='nrmse' and boundary<=0):
                    raise ValueError(f'Undefined or vacuous evaluator boundary: {name}')
                report['thresholds'][name]={'comparison':comparison,'boundary':boundary}
        truth=assets/'ground_truth.npz'
        reference=load_arrays(truth if truth.is_file() else assets/'baseline_reference.npz')
        for key,r in reference.items():
            if np.ptp(r)<=0 or np.linalg.norm(r)==0:
                raise ValueError(f'Degenerate evaluator reference: {key}')
        arrays=load_arrays(prediction)
        measured={}
        for key in KEYS:
            x=arrays[key].ravel();r=reference[key].ravel()
            denom=np.linalg.norm(x)*np.linalg.norm(r)
            measured[f'{key}_ncc']=float(x@r/denom) if denom else 0.
            measured[f'{key}_nrmse']=float(np.sqrt(np.mean((x-r)**2))/np.ptp(r))
        if not all(math.isfinite(x) for x in measured.values()):
            raise ValueError('Evaluation produced a nonfinite metric')
        passed=all(value>=report['thresholds'][name]['boundary'] if name.endswith('_ncc')
                   else value<=report['thresholds'][name]['boundary'] for name,value in measured.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(Path(args.prediction))
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr)
        return 2
    return code


if __name__=='__main__':
    raise SystemExit(main())
