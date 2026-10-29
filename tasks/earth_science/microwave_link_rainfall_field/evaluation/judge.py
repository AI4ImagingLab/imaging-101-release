"""Independent full-field rainfall evaluator; no solver imports or execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent

def load_field(path):
 """Strict physical output contract: real float32/64, full batch-first field."""
 if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
 with np.load(path,allow_pickle=False) as z:
  if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
  a=z['reconstruction']
 if a.shape!=(1,38063) or a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64(1,38063)')
 if not np.isfinite(a).all():raise ValueError('Nonfinite rainfall field')
 return a.astype(np.float64)

def field_error(estimate,reference):
 """RMS full-field error divided by reference dynamic range, in identical mm units."""
 with np.errstate(over='raise',invalid='raise',divide='raise'):
  span=float(np.max(reference)-np.min(reference))
  if not np.isfinite(span) or span<=0:raise ValueError('Degenerate hidden reference range')
  error=float(np.sqrt(np.mean(((estimate-reference)/span)**2)))
 if not np.isfinite(error):raise ValueError('Metric overflow')
 return error

def evaluate(prediction,evaluation_dir=None):
 root=Path(evaluation_dir) if evaluation_dir is not None else HERE
 report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
 try:
  config=json.loads((root/'metrics.json').read_text(encoding='utf8'))
  for name in ['nrmse_boundary','negative_count_boundary']:
   value=config[name]
   if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError('Invalid nonnegative finite numeric threshold '+name)
  if config['negative_count_boundary']!=int(config['negative_count_boundary']) or config['negative_count_boundary']>38063:raise ValueError('Invalid negative-count threshold')
  reference=root/'ground_truth.npz'
  if not reference.exists():reference=root/'baseline_reference.npz'
  truth=load_field(reference);estimate=load_field(prediction)
  metrics=dict(nrmse=field_error(estimate,truth),negative_count=int(np.sum(estimate<0)))
  thresholds={key:dict(comparison='<=',boundary=config[key+'_boundary']) for key in metrics}
  passed=all(metrics[k]<=t['boundary'] for k,t in thresholds.items())
  report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
 except Exception as error:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(error)])
 return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
 try:
  path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as error:print('Report write failure: '+str(error),file=sys.stderr);return 2
 return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2

if __name__=='__main__':raise SystemExit(main())
