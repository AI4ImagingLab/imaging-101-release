"""Local Gaussian/RPL/cold-plasma acquisition mathematics; source license in NOTICE."""
import numpy as np
from scipy.special import erf
K=4149.377593360996;NOMINAL_DM=623.6785965856119;REF_FREQ=400.1953125

def gaussian_model(q,frequencies,times,ff=8,ft=4,analytic_time=False):
 logamp,t0,width,dm,index,running=q;df=frequencies[1]-frequencies[0];dt=times[1]-times[0];out=np.empty((len(frequencies),len(times)))
 ts=np.linspace(times[0]-dt/2+dt/(2*ft),times[-1]+dt/2-dt/(2*ft),len(times)*ft)
 for lo in range(0,len(frequencies),128):
  fc=frequencies[lo:lo+128];sub=np.linspace(fc-df/2+df/(2*ff),fc+df/2-df/(2*ff),ff,axis=-1)
  arrival=t0+(NOMINAL_DM+dm)*K*(sub**-2-REF_FREQ**-2)-NOMINAL_DM*K*(fc[:,None]**-2-REF_FREQ**-2)
  lognu=np.log(sub/REF_FREQ);spectral=np.exp(index*lognu+running*lognu**2)
  if analytic_time:
   upper=(times[None,None,:]+dt/2-arrival[:,:,None])/(np.sqrt(2)*width);lower=upper-dt/(np.sqrt(2)*width)
   p=np.sqrt(np.pi/2)*width/dt*(erf(upper)-erf(lower));result=(p*spectral[:,:,None]).mean(1)
  else:
   p=np.exp(-.5*((ts[None,None,:]-arrival[:,:,None])/width)**2)*spectral[:,:,None]
   result=p.mean(1).reshape(len(fc),len(times),ft).mean(2)
  out[lo:lo+len(fc)]=10**logamp*result
 return out

import argparse,json,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();report=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
 try:
  here=Path(__file__).resolve().parent;config=json.loads((here/'metrics.json').read_text(encoding='utf8'));thresholds=config.get('thresholds')
  if not isinstance(thresholds,dict) or set(thresholds)!={'weighted_mean_squared_residual'}:raise ValueError('invalid evaluator metric set')
  item=thresholds['weighted_mean_squared_residual']
  if not isinstance(item,dict) or set(item)!={'comparison','boundary'} or item['comparison']!='<=':raise ValueError('invalid evaluator comparison')
  bound=item['boundary']
  if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not np.isfinite(bound) or bound<=0:raise ValueError('invalid evaluator boundary')
  report['thresholds']=thresholds
  with np.load(a.prediction,allow_pickle=False) as f:
   if 'reconstruction' not in f:raise ValueError('missing reconstruction')
   q=f['reconstruction']
   if q.shape!=(1,6):raise ValueError('expected shape (1,6)')
   if q.dtype!=np.float64:raise ValueError('expected real float64')
   if not np.isfinite(q).all():raise ValueError('nonfinite prediction')
   q=q[0]
  with np.load(here.parent/'data/raw_data.npz',allow_pickle=False) as f:d={k:f[k][0] for k in f.files}
  if q[2]<=0 or q[3]+NOMINAL_DM<=0:report.update(status='fail',errors=['require positive intrinsic width and absolute dispersion measure']);code=1
  else:
   model=gaussian_model(q,d['frequencies'],d['times']);good=d['good_channels'];r=(d['measurements'][good]-model[good])/d['noise_rms'][good,None]
   value=float(np.mean(r*r))
   if not np.isfinite(value):report.update(status='fail',errors=['nonfinite physical model for finite submitted parameters']);code=1
   else:
    passed=value<=bound;report.update(status='pass' if passed else 'fail',passed=passed,metrics={'weighted_mean_squared_residual':value});code=0 if passed else 1
 except Exception as e:report['errors']=[str(e)];report['metrics']={}
 try:
  p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failed: '+str(e).encode('ascii','backslashreplace').decode(),file=sys.stderr);return 2
 return code
if __name__=='__main__':raise SystemExit(main())
