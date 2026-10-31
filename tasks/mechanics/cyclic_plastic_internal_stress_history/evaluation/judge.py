"""Uniaxial four-component MAFTr closed forms; see NOTICE/source provenance."""
import numpy as np

def internal_states(parameters,plastic_strain):
 q=np.asarray(parameters,dtype=float);eps=np.asarray(plastic_strain,dtype=float)
 k0,k1,k2=q[:3];c=q[3:7];a=q[7:]*np.sqrt(2/3);bar=a[4]
 D=np.r_[0.,np.sign(np.diff(eps))];acc=np.r_[0.,np.cumsum(abs(np.diff(eps)))];kiso=D/k2*(1-(1-k0*k2)*np.exp(-np.sqrt(1.5)*k1*k2*acc))
 ends=np.r_[np.flatnonzero(D[:-1]*D[1:]==-1),len(eps)-1]
 alpha=np.zeros((4,len(eps)));ref=np.zeros(4);epsref=eps[0];lo=0
 for hi in ends:
  sl=slice(lo,hi+1);direction=D[sl];delta=abs(eps[sl]-epsref)
  alpha[:3,sl]=direction[None,:]*a[:3,None]-(direction[None,:]*a[:3,None]-ref[:3,None])*np.exp(-c[:3,None]*delta[None,:])
  g=np.full(len(delta),ref[3]);remaining=delta.copy();negative=direction*ref[3]<-bar;tn=np.zeros(len(delta))
  if np.any(negative):
   tn[negative]=-np.log(a[3]/(a[3]-bar-direction[negative]*ref[3]))/c[3]
   step=np.minimum(remaining[negative],tn[negative]);g[negative]=direction[negative]*(a[3]-bar)-(direction[negative]*(a[3]-bar)-ref[3])*np.exp(-c[3]*step)
   remaining[negative]=np.maximum(remaining[negative]-tn[negative],0.)
  linear=(remaining>0)&((tn>0)|(direction*g<=bar))
  if np.any(linear):
   tl=(bar-direction[linear]*g[linear])/(c[3]*a[3]);g[linear]+=direction[linear]*c[3]*a[3]*np.minimum(remaining[linear],tl);remaining[linear]=np.maximum(remaining[linear]-tl,0.)
  positive=remaining>0
  g[positive]=direction[positive]*(a[3]+bar)-(direction[positive]*(a[3]+bar)-g[positive])*np.exp(-c[3]*remaining[positive])
  alpha[3,sl]=g;ref=alpha[:,hi].copy();epsref=eps[hi];lo=hi+1
 return alpha.T,kiso,np.sqrt(1.5)*alpha.sum(0)+kiso

import argparse,json,sys
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
 try:
  here=Path(__file__).resolve().parent;config=json.loads((here/'metrics.json').read_text(encoding='utf8'))
  expected={'stress_mse':'<=','backstress_model_error':'<=','isotropic_model_error':'<=','minimum_parameter':'>'}
  thresholds=config.get('thresholds')
  if not isinstance(thresholds,dict) or set(thresholds)!=set(expected):raise ValueError('invalid evaluator threshold metric set')
  for key,comparison in expected.items():
   item=thresholds[key]
   if not isinstance(item,dict) or set(item)!={'comparison','boundary'} or item['comparison']!=comparison:raise ValueError('invalid evaluator comparison for '+key)
   bound=item['boundary']
   if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not np.isfinite(bound) or bound<0 or (key!='minimum_parameter' and bound<=0):raise ValueError('invalid evaluator boundary for '+key)
  report['thresholds']=thresholds
  with np.load(args.prediction,allow_pickle=False) as f:
   arrays={}
   for key,shape in [('parameters',(1,12)),('reconstruction',(1,1003,4)),('isotropic_stress',(1,1003))]:
    if key not in f:raise ValueError('missing '+key)
    x=f[key]
    if x.shape!=shape:raise ValueError('wrong shape '+key)
    if x.dtype!=np.float64:raise ValueError('expected real float64 '+key)
    if not np.isfinite(x).all():raise ValueError('nonfinite '+key)
    arrays[key]=x[0]
  with np.load(here.parent/'data/raw_data.npz',allow_pickle=False) as raw:eps=raw['plastic_strain'][0];stress=raw['measured_stress'][0]
  q=arrays['parameters'];minimum=float(q.min())
  if minimum<=0:
   report['metrics']={'minimum_parameter':minimum};report['status']='fail';code=1
  else:
   alpha,kiso,pred=internal_states(q,eps)
   outputstress=np.sqrt(1.5)*arrays['reconstruction'].sum(1)+arrays['isotropic_stress']
   m=dict(stress_mse=float(np.mean((outputstress-stress)**2)),backstress_model_error=float(np.max(abs(alpha-arrays['reconstruction']))/max(1.,np.max(abs(alpha)))),isotropic_model_error=float(np.max(abs(kiso-arrays['isotropic_stress']))/max(1.,np.max(abs(kiso)))),minimum_parameter=minimum)
   if not all(np.isfinite(v) for v in m.values()):raise ValueError('nonfinite constitutive model')
   report['metrics']=m;report['passed']=all(m[k]<=v['boundary'] if v['comparison']=='<=' else m[k]>v['boundary'] for k,v in report['thresholds'].items());report['status']='pass' if report['passed'] else 'fail';code=0 if report['passed'] else 1
 except Exception as e:report['errors']=[str(e)];report['metrics']={}
 try:
  p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failed: '+str(e),file=sys.stderr);return 2
 return code
if __name__=='__main__':sys.exit(main())

