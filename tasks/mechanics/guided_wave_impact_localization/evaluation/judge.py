"""Independent controlled-impact coordinate evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
CASES=('HEI','ANG','TEM')
def coordinate_metrics(prediction,truth):
 distance=np.linalg.norm(prediction[0]-truth[0],axis=-1);out={}
 for c,d in zip(CASES,distance):
  out[c+'_rmse_mm']=float(np.sqrt(np.mean(d*d)));out[c+'_p90_mm']=float(np.quantile(d,.9,method='linear'));out[c+'_maximum_mm']=float(d.max())
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();r=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
 try:
  here=Path(__file__).resolve().parent;cfg=json.loads((here/'metrics.json').read_text(encoding='utf8'));ts=cfg.get('thresholds');expected={c+'_'+m+'_mm' for c in CASES for m in ('rmse','p90','maximum')}
  if not isinstance(ts,dict) or set(ts)!=expected:raise ValueError('invalid evaluator metric set')
  for v in ts.values():
   if not isinstance(v,dict) or set(v)!={'comparison','boundary'} or v['comparison']!='<=':raise ValueError('invalid evaluator comparison')
   b=v['boundary']
   if isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(b) or b<=0:raise ValueError('invalid evaluator boundary')
  r['thresholds']=ts
  def read(path):
   with np.load(path,allow_pickle=False) as f:
    if 'reconstruction' not in f:raise ValueError('missing reconstruction')
    z=f['reconstruction']
    if z.shape!=(1,3,35,2) or z.dtype!=np.float64 or not np.isfinite(z).all():raise ValueError('require finite float64 (1,3,35,2)')
    return z
  z=read(a.prediction);truth=read(here/'ground_truth.npz');m=coordinate_metrics(z,truth)
  inside=bool(np.all((z[...,0]>=0)&(z[...,0]<=160)&(z[...,1]>=0)&(z[...,1]<=120)))
  passed=inside and all(m[k]<=v['boundary'] for k,v in ts.items());r.update(status='pass' if passed else 'fail',passed=passed,metrics=m);code=0 if passed else 1
  if not inside:r['errors']=['coordinates outside public instrumented rectangle']
 except Exception as e:r.update(status='error',passed=False,metrics={},errors=[str(e)]);code=2
 try:
  p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
 except Exception as e:print('report write failed: '+str(e).encode('ascii','backslashreplace').decode(),file=sys.stderr);return 2
 return code
if __name__=='__main__':raise SystemExit(main())
