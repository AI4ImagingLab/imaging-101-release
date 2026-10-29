"""Standalone evaluator of all three signed covariance components and transmission."""
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
        if 'reconstruction' not in z or 'transmission' not in z:raise ValueError('required reconstruction and transmission keys are missing')
        c=z['reconstruction'];t=z['transmission']
    for a,shape,name in [(c,(1,3,35,35),'reconstruction'),(t,(1,35,35),'transmission')]:
        if a.shape!=shape:raise ValueError(f'{name} shape must be {shape}')
        if a.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError(f'{name} dtype must be float32 or float64')
        if not np.isfinite(a).all():raise ValueError(f'{name} contains nonfinite values')
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:r=z['reconstruction'];rt=z['transmission'];observed=z['identified_mode_ratios']
    for a,shape in [(r,(1,3,35,35)),(rt,(1,35,35)),(observed,(1,5,35,35))]:
        if a.shape!=shape or a.dtype not in [np.dtype('float32'),np.dtype('float64')] or not np.isfinite(a).all():raise ValueError('invalid evaluator asset')
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));metrics={};bounds={}
    for name,a,b in [('cxx',c[:,0],r[:,0]),('cyy',c[:,1],r[:,1]),('cxy',c[:,2],r[:,2]),('transmission',t,rt)]:
        values=compare(a,b)
        for key,value in values.items():
            full=name+'_'+key;metrics[full]=value;bounds[full]={'comparison':'>=' if key=='ncc' else '<=','boundary':config[full+'_boundary']}
    if max(float(np.max(abs(c))),float(np.max(abs(t))))>1e100:residual=1e300
    else:
        xx,yy,xy=c[0];projected=np.array([xx,yy,xx+yy-2*xy,xx+yy+2*xy]);attenuation=np.exp(np.clip(-2*np.pi**2/12.68550106609808**2*projected,-700,700));pred=np.concatenate([t[0,None],t[0,None]*attenuation]);residual=float(min(1e300,np.linalg.norm(pred-observed[0])/np.linalg.norm(observed)))
    metrics['mode_residual']=residual;bounds['mode_residual']={'comparison':'<=','boundary':config['mode_residual_boundary']}
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
