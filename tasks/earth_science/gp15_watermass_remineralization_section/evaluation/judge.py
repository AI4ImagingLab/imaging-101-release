import numpy as np
NAMES=('AAIW','PSUW','ESSW','NPIW','PDW1','UCDW','PDW23','LCDW','AABW')
def usage_penalties(coordinates):
 lat,lon,depth,density=np.asarray(coordinates).T
 p=np.zeros((len(lat),9));densitypen=np.minimum(10000,1000*np.expm1(.5*np.maximum(27.72-density,0)))
 p[:,7]=densitypen;p[:,8]=densitypen
 p[:,1]=np.minimum(10000,100*np.expm1(.05*np.maximum(10-lat,0)))+np.minimum(10000,50*np.expm1(.001*np.maximum(depth-3000,0)))
 return p

def predict(fractions,reactions,endmembers,ratios):return fractions@endmembers+reactions@ratios

def objective_gradient(q,observation,endmembers,ratios,weights,penalty):
 n=len(endmembers);a=np.vstack([endmembers,ratios]);res=(q@a-observation)*weights
 value=np.dot(res,res)+np.sum((q[:n]*penalty)**2)
 grad=2*a@(res*weights);grad[:n]+=2*q[:n]*penalty**2
 return float(value),grad

import argparse,json,sys
from pathlib import Path

def losses(f,r,d):
 pred=f@d['endmembers']+r@d['reaction_ratios'];p=usage_penalties(d['sample_coordinates'])
 return np.sum(((pred-d['observations'])*d['tracer_weights'])**2,axis=1)+np.sum((f*p)**2,axis=1)
def main():
 a=argparse.ArgumentParser();a.add_argument('--prediction',required=True);a.add_argument('--report',required=True);args=a.parse_args();result=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
 try:
  here=Path(__file__).resolve().parent;config=json.loads((here/'metrics.json').read_text(encoding='utf8'))
  expected={'maximum_normalized_objective_excess': '<=', 'maximum_simplex_error': '<=', 'minimum_fraction': '>=', 'minimum_remineralization': '>='}
  thresholds=config.get('thresholds')
  if not isinstance(thresholds,dict) or set(thresholds)!=set(expected):raise ValueError('invalid evaluator threshold metric set')
  for key,comparison in expected.items():
   item=thresholds[key]
   if not isinstance(item,dict) or set(item)!={'boundary','comparison'} or item['comparison']!=comparison:raise ValueError('invalid evaluator comparison for '+key)
   bound=item['boundary']
   if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not np.isfinite(bound):raise ValueError('invalid evaluator finite boundary for '+key)
   if (comparison=='<=' and bound<=0) or (comparison=='>=' and bound>0):raise ValueError('invalid evaluator boundary sign for '+key)
  result['thresholds']=thresholds
  def read(path,validate=True):
   with np.load(path,allow_pickle=False) as data:
    out=[]
    for key,shape in [('reconstruction',(1,576,9)),('remineralization',(1,576,2))]:
     if key not in data:raise ValueError('missing '+key)
     x=data[key]
     if x.shape!=shape:raise ValueError('wrong shape for '+key)
     if x.dtype!=np.float64:raise ValueError('expected real float64 '+key)
     if not np.isfinite(x).all():raise ValueError('nonfinite '+key)
     out.append(x[0])
    return out
  f,r=read(args.prediction);bf,br=read(here/'baseline_reference.npz')
  with np.load(here.parent/'data/raw_data.npz',allow_pickle=False) as raw:d={k:raw[k][0] for k in raw.files}
  v=losses(f,r,d);ref=losses(bf,br,d)
  m=dict(maximum_normalized_objective_excess=float(np.max((v-ref)/np.maximum(1,ref))),maximum_simplex_error=float(np.max(np.abs(f.sum(1)-1))),minimum_fraction=float(f.min()),minimum_remineralization=float(r.min()))
  if not all(np.isfinite(v) for v in m.values()):raise ValueError('nonfinite evaluator metric')
  result['metrics']=m;result['passed']=all(m[k]<=v['boundary'] if v['comparison']=='<=' else m[k]>=v['boundary'] for k,v in result['thresholds'].items());result['status']='pass' if result['passed'] else 'fail';code=0 if result['passed'] else 1
 except Exception as e:result['errors']=[str(e)];result['metrics']={}
 try:
  p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failure: '+str(e),file=sys.stderr);return 2
 return code
if __name__=='__main__':sys.exit(main())
