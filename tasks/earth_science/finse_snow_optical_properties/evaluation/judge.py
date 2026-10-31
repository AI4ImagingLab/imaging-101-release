"""Explicit dense/ELU forward response and input-gradient chain rule; GPL-3.0."""
import numpy as np

def forward(normalized, data, target=None):
 x=np.asarray(normalized,dtype=np.float32);states=[]
 for i in range(12):
  z=x@data[f'weight_{i}']+data[f'bias_{i}'];states.append(z)
  x=z if i==11 else np.where(z>0,z,np.expm1(np.minimum(z,0)))
 if target is None:return x
 residual=x[:,6:106]-target
 g=np.zeros_like(x);g[:,6:106]=residual
 for i in range(11,-1,-1):
  if i<11:g*=np.where(states[i]>0,1,np.exp(np.minimum(states[i],0)))
  g=g@data[f'weight_{i}'].T
 g=g.astype(np.float64);g[:,0]=0
 return residual,g

def normalized_parameters(parameters,data):
 physical=np.c_[data['solar_zenith'],parameters]
 return physical*data['scaler_scale']+data['scaler_offset']

def spectrum(parameters,data):return forward(normalized_parameters(parameters,data),data)

import argparse,json,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();report=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
 try:
  here=Path(__file__).resolve().parent
  config=json.loads((here/'metrics.json').read_text(encoding='utf8'))
  expected={'mean_absolute_spectral_error','worst_spectrum_absolute_error','forward_consistency_max'}
  thresholds=config.get('thresholds')
  if not isinstance(thresholds,dict) or set(thresholds)!=expected:raise ValueError('invalid evaluator metric set')
  for key,item in thresholds.items():
   if not isinstance(item,dict) or set(item)!={'boundary','comparison'} or item['comparison']!='<=':raise ValueError('invalid evaluator comparison '+key)
   bound=item['boundary']
   if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not np.isfinite(bound) or bound<=0:raise ValueError('invalid evaluator boundary '+key)
  report['thresholds']=thresholds
  with np.load(here.parent/'data/raw_data.npz',allow_pickle=False) as f:data={k:f[k][0] for k in f.files}
  with np.load(a.prediction,allow_pickle=False) as f:
   arrays={}
   for key,shape,dtype in [('reconstruction',(1,180,5),np.float64),('modeled_spectra',(1,180,212),np.float32)]:
    if key not in f:raise ValueError('missing '+key)
    x=f[key]
    if x.shape!=shape:raise ValueError('wrong shape '+key)
    if x.dtype!=dtype:raise ValueError('wrong dtype '+key)
    if not np.all(np.isfinite(x)):raise ValueError('nonfinite '+key)
    arrays[key]=x[0]
  q=arrays['reconstruction'];z=arrays['modeled_spectra'];pred=spectrum(q,data)
  if np.any(q[:,0]<350) or np.any(q[:,1:]<0) or np.any(q[:,2]>1):
   report.update(status='fail',errors=['physical domain: radius>=350micrometre and all concentrations/water content>=0']);code=1
  else:
   err=np.mean(abs(z[:,6:106]-data['spectra'][:,5::10][:,:100]),1)
   metrics={'mean_absolute_spectral_error':float(err.mean()),'worst_spectrum_absolute_error':float(err.max()),'forward_consistency_max':float(np.max(abs(pred-z)))}
   passed=all(metrics[k]<=v['boundary'] for k,v in config['thresholds'].items())
   report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics);code=0 if passed else 1
 except Exception as e:report['errors'].append(str(e))
 try:
  p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failed: '+str(e).encode('ascii','backslashreplace').decode('ascii'),file=sys.stderr);return 2
 return code
if __name__=='__main__':raise SystemExit(main())
