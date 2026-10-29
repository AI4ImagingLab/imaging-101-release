"""Standalone numerical/observation-consistency evaluator; never imports src."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.ndimage import maximum_filter,minimum_filter
HERE=Path(__file__).resolve().parent;SHAPE=(1,512,512);DZ=2.358/511
def load_array(path,role):
 if not Path(path).is_file():raise ValueError(f'{role}: missing NPZ {path}')
 with np.load(path,allow_pickle=False) as d:
  if 'reconstruction' not in d:raise ValueError(f'{role}: missing reconstruction key')
  a=d['reconstruction']
 if a.shape!=SHAPE:raise ValueError(f'{role}: expected shape{SHAPE},got{a.shape}')
 if a.dtype!=np.dtype('float64') or np.iscomplexobj(a):raise ValueError(f'{role}: require realfloat64')
 if not np.isfinite(a).all():raise ValueError(f'{role}: NaN/Inf values')
 if np.min(a)<0 or np.max(a)>(1/(2*DZ))*(1+1e-12):raise ValueError(f'{role}: outside declared nonnegative discrete rate range')
 return a
def numerical_metrics(estimate,reference):
 r=np.ptp(reference)
 if not r>0:raise ValueError('Evaluator reference must have nonzero dynamic range')
 a=estimate.ravel();b=reference.ravel();na=np.linalg.norm(a);nb=np.linalg.norm(b)
 if not nb>0:raise ValueError('Evaluator reference has zero norm')
 ncc=float((a/na)@(b/nb)) if na>0 else 0.
 nrmse=float(np.sqrt(np.mean((a-b)**2))/r)
 return dict(ncc=ncc,nrmse=nrmse)
def observation_consistency(estimate):
 raw=HERE.parent/'data/raw_data.npz'
 if not raw.is_file():raise ValueError('Evaluator public observations unavailable')
 with np.load(raw,allow_pickle=False) as d:a=d['oct_signal']
 if a.shape!=SHAPE or a.dtype!=np.dtype('float64') or not np.isfinite(a).all() or a.min()<0:raise ValueError('Evaluator observation asset invalid')
 a=a[0];s=np.sort(a,axis=0);threshold=(s[:51].mean(axis=0)+s[460:].mean(axis=0))/2
 crossing=np.diff((a>threshold).astype(np.int8),axis=0)==1
 loc=np.where(np.any(crossing,axis=0),np.argmax(crossing,axis=0)+1,0)
 mask=(np.arange(512)[:,None]>=loc[None,:]).astype(np.uint8)
 mask=minimum_filter(maximum_filter(mask,size=10,mode='constant',cval=0),size=35,mode='constant',cval=1)
 a=a*mask;z=np.linspace(0,2.358,512)
 h=1/(1+((z-1.)/.29017970000000004)**2);zeta=np.pi/2*z/(1.325e-6**2/(4*7.1875e-11)*1000)
 f=np.sinc(zeta/np.pi)**2*np.exp(-(1.1055e-10/7.1875e-11)**2/(2*np.log(2))*zeta**2)
 term=a/(h*f)[:,None]*(a*a/(a*a+2.5e-5));cumulative=np.cumsum(term[:-1][::-1],axis=0)[::-1]
 residual=term[:-1]/(2*DZ)-estimate[0,:-1]*cumulative
 norm=np.linalg.norm(term[:-1]/(2*DZ))
 if norm<=0:raise ValueError('Evaluator observation norm is zero')
 return float(np.linalg.norm(residual)/norm)
def evaluate(prediction):
 report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
 try:
  settings=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
  thresholds={'ncc':dict(comparison='>=',boundary=float(settings['ncc_boundary'])),'nrmse':dict(comparison='<=',boundary=float(settings['nrmse_boundary'])),'forward_residual':dict(comparison='<=',boundary=float(settings['forward_residual_boundary']))}
  if not all(np.isfinite(v['boundary']) for v in thresholds.values()) or not -1<=thresholds['ncc']['boundary']<=1 or thresholds['nrmse']['boundary']<0 or thresholds['forward_residual']['boundary']<0:raise ValueError('Evaluator metric definitions invalid')
  report['thresholds']=thresholds
  estimate=load_array(prediction,'Prediction')
  reference_path=HERE/'ground_truth.npz'
  if not reference_path.is_file():reference_path=HERE/'baseline_reference.npz'
  reference=load_array(reference_path,'Evaluator reference')
  metrics=numerical_metrics(estimate,reference);metrics['forward_residual']=observation_consistency(estimate)
  if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Evaluator metrics are not finite')
  passed=metrics['ncc']>=thresholds['ncc']['boundary'] and metrics['nrmse']<=thresholds['nrmse']['boundary'] and metrics['forward_residual']<=thresholds['forward_residual']['boundary']
  report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
  return report,0 if passed else 1
 except Exception as e:report['metrics']={};report['errors']=[str(e)];return report,2
def main():
 p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
 report,code=evaluate(Path(args.prediction))
 try:
  path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print(f'Report-write failure: {e}',file=sys.stderr);return 2
 print(json.dumps(report,allow_nan=False));return code
if __name__=='__main__':raise SystemExit(main())
