"""Independent full-field physical-unit Judge; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def evaluate(prediction,reference_dir=None):
    directory=ROOT if reference_dir is None else Path(reference_dir)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        reference=directory/'ground_truth.npz'
        if not reference.exists():reference=directory/'baseline_reference.npz'
        def load(path):
            with np.load(path,allow_pickle=False) as z:
                if 'reconstruction' not in z:raise ValueError('Required key reconstruction is missing')
                a=z['reconstruction']
            if a.shape!=(1,4,300,480):raise ValueError('reconstruction must have shape (1,4,300,480)')
            if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('reconstruction must be real float32 or float64')
            if not np.isfinite(a).all():raise ValueError('reconstruction contains NaN or Infinity')
            return a
        ref=load(reference);estimate=load(Path(prediction))
        names=('remanent_B','bias_odd_B','D_configuration0','D_configuration1')
        diff=estimate.astype('float64')-ref.astype('float64');passed=True
        for i,name in enumerate(names):
            kind='B' if i<2 else 'D'
            maximum=float(np.max(abs(diff[0,i])))
            rmse=0. if maximum==0 else maximum*float(np.sqrt(np.mean((diff[0,i]/maximum)**2)))
            for metric,value in [('rmse',rmse),('max_abs',maximum)]:
                raw_boundary=config[f'{kind}_{metric}_boundary']
                if type(raw_boundary) not in (int,float):raise ValueError('Metric boundary must be an actual JSON number, not bool or string')
                boundary=float(raw_boundary)
                if not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid metric boundary')
                key=f'{name}_{metric}';report['metrics'][key]=value
                report['thresholds'][key]={'comparison':'<=','boundary':boundary}
                passed=passed and value<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed));return report
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[f'{type(exc).__name__}: {exc}']);return report
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())

