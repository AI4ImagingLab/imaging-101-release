"""Standalone evaluator of complete equilibrium energy, thermodynamics and baselines."""
from pathlib import Path
import sys,json,argparse
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,15,3181),'thermodynamic_parameters':(1,4),'baseline_coefficients':(1,2,7)}
def load_arrays(path):
 if not Path(path).is_file():raise ValueError('File not found: '+str(path))
 with np.load(path,allow_pickle=False) as z:
  out={}
  for key,shape in SHAPES.items():
   if key not in z:raise ValueError('Missing '+key)
   a=z[key]
   if a.shape!=shape:raise ValueError('Wrong shape '+key)
   if a.dtype!=np.dtype('float64'):raise ValueError('Expected float64 '+key)
   if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
   out[key]=a[0]
 p=out['thermodynamic_parameters']
 if p[0]<=-273.15 or np.any(p[1:]<=0):raise ValueError('Invalid physical thermodynamic orientation or Kelvin midpoint')
 return out
def evaluate(prediction,evaluation_dir=None):
 directory=HERE if evaluation_dir is None else Path(evaluation_dir);result=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
 try:
  cfg=json.loads((directory/'metrics.json').read_text(encoding='utf8'));names=['energy_max_abs','thermodynamic_scaled_max_abs','baseline_max_abs','observation_relative_l2','energy_internal_consistency'];rules={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in names}
  if not all(isinstance(v['boundary'],(int,float)) and np.isfinite(v['boundary']) and v['boundary']>0 for v in rules.values()):raise ValueError('Invalid evaluator boundary')
  reference=directory/'ground_truth.npz'
  if not reference.exists():reference=directory/'baseline_reference.npz'
  ref=load_arrays(reference);pred=load_arrays(prediction)
  with np.load(directory.parent/'data/raw_data.npz',allow_pickle=False) as z:t=z['temperature_celsius'][0];d=z['denaturant_molar'][0];y=z['intensity'][0]
  if t.shape!=(15,3181) or d.shape!=(15,) or y.shape!=(2,15,3181) or not all(np.isfinite(a).all() for a in [t,d,y]):raise ValueError('Malformed acquisition')
  indices=np.argmin(abs(t-70.),axis=1);scale=y[:,np.arange(15),indices]
  if np.any(scale==0) or np.any(t<=-273.15):raise ValueError('Invalid optical scale or absolute temperature')
  observed=y/scale[:,:,None];tm,dh,cp,m=pred['thermodynamic_parameters'];K=t+273.15;M=tm+273.15;energy=dh*(1-K/M)+cp*(K-M-K*np.log(K/M))-d[:,None]*m
  with np.errstate(over='ignore',invalid='ignore'):
   odds=np.exp(-np.maximum(energy,-250.)/(.008314*K));u=odds/(1+odds);model=np.empty_like(observed);s=t-25.
   for c in range(2):
    aF,bFD,bFT,aU,bUD,bUT,bUT2=pred['baseline_coefficients'][c];F=aF+bFD*d[:,None]+bFT*s;U=aU+bUD*d[:,None]+bUT*s+bUT2*s*s;model[c]=(1-u)*F+u*U
  denominator=np.linalg.norm(observed)
  if denominator==0:raise ValueError('Undefined observation norm')
  metrics=dict(energy_max_abs=float(np.max(abs(pred['reconstruction']-ref['reconstruction']))),thermodynamic_scaled_max_abs=float(np.max(abs(pred['thermodynamic_parameters']-ref['thermodynamic_parameters'])/np.maximum(1,abs(ref['thermodynamic_parameters'])))),baseline_max_abs=float(np.max(abs(pred['baseline_coefficients']-ref['baseline_coefficients']))),observation_relative_l2=float(np.linalg.norm(model-observed)/denominator),energy_internal_consistency=float(np.max(abs(pred['reconstruction']-energy))))
  if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Numerically undefined evaluator metric')
  passed=all(metrics[k]<=v['boundary'] for k,v in rules.items());result.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=rules);return result,0 if passed else 1
 except Exception as exc:result['errors']=[str(exc)];return result,2
def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv);report,code=evaluate(args.prediction)
 try:
  p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
 return code
if __name__=='__main__':sys.exit(main())
