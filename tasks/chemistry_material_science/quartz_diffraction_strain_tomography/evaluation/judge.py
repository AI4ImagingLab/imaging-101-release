"""Independent six-component actual-truth evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def compare(estimate,reference):
    ep=float(np.max(abs(estimate)));rp=float(np.max(abs(reference)))
    if rp==0 or np.ptp(reference)<=0:raise ValueError('degenerate evaluator reference')
    if ep==0:return {'ncc':0.,'nrmse':float(np.sqrt(np.mean(reference.astype(float)**2))/np.ptp(reference))}
    e=estimate.astype(float)/ep;r=reference.astype(float)/rp
    ncc=float(np.vdot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r)))
    common=max(ep,rp);error=e*(ep/common)-r*(rp/common)
    return {'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(min(1e300,np.sqrt(np.mean(error**2))*min(common/float(np.ptp(reference)),1e300)))}

def evaluate(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('missing reconstruction key')
        a=z['reconstruction']
    if a.shape!=(1,6,106,96):raise ValueError('reconstruction shape must be (1,6,106,96)')
    if a.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError('reconstruction must be real float32 or float64')
    if not np.isfinite(a).all():raise ValueError('nonfinite reconstruction')
    with np.load(HERE/'ground_truth.npz',allow_pickle=False) as z:r=z['reconstruction']
    if r.shape!=a.shape or not np.isfinite(r).all() or r.dtype!=np.float64:raise ValueError('invalid evaluator truth asset')
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));metrics={};bounds={}
    for i,key in enumerate(['xx','yy','zz','xy','xz','yz']):
        for metric,value in compare(a[:,i],r[:,i]).items():
            name=key+'_'+metric;boundary=config[name+'_boundary']
            if not np.isfinite(boundary):raise ValueError('nonfinite evaluator boundary')
            metrics[name]=value;bounds[name]={'comparison':'>=' if metric=='ncc' else '<=','boundary':boundary}
    passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in bounds.items())
    return {'status':'pass' if passed else 'fail','passed':bool(passed),'metrics':metrics,'thresholds':bounds,'errors':[]}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    try:result=evaluate(Path(args.prediction));code=0 if result['passed'] else 1
    except Exception as exc:result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print(f'report write failed: {exc}',file=sys.stderr);return 2
    print(json.dumps(result,allow_nan=False));return code
if __name__=='__main__':raise SystemExit(main())
