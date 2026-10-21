"""Standalone evaluator, with an independent ray/pixel segment-intersection matrix."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
HERE=Path(__file__).resolve().parent
SHAPE=(1,128,128,128)
def compare(estimate,reference):
    ep=float(np.max(abs(estimate)));rp=float(np.max(abs(reference)))
    if rp==0 or np.ptp(reference)<=0:raise ValueError('degenerate evaluator reference')
    if ep==0:return {'ncc':0.,'nrmse':float(np.sqrt(np.mean(reference.astype(float)**2))/np.ptp(reference))}
    e=estimate.astype(float)/ep;r=reference.astype(float)/rp
    ncc=float(np.vdot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r)))
    common=max(ep,rp);error=e*(ep/common)-r*(rp/common)
    return {'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(min(1e300,np.sqrt(np.mean(error**2))*min(common/float(np.ptp(reference)),1e300)))}
def ray_matrix(n,angles):
    edges=np.arange(n+1,dtype=float)-n/2; rows=[];cols=[];values=[]
    for ai,angle in enumerate(angles):
        theta=np.deg2rad(angle);direction=np.array([-np.sin(theta),np.cos(theta)]);normal=np.array([np.cos(theta),np.sin(theta)])
        for ray,offset in enumerate(np.arange(n)-(n-1)/2):
            origin=offset*normal;ts=[]
            for ax in range(2):
                if abs(direction[ax])>1e-12:ts.extend((edges-origin[ax])/direction[ax])
            ts=np.unique(np.sort(ts));mid=(ts[:-1]+ts[1:])/2;xy=origin[:,None]+direction[:,None]*mid
            col=np.floor(xy[0]+n/2).astype(int);row=np.floor(n/2-xy[1]).astype(int)
            good=(row>=0)&(row<n)&(col>=0)&(col<n)
            rows.extend([ai*n+ray]*int(good.sum()));cols.extend((row[good]*n+col[good]).tolist());values.extend(np.diff(ts)[good].tolist())
    return coo_matrix((np.asarray(values,dtype='float32'),(rows,cols)),shape=(len(angles)*n,n*n)).tocsr()
def evaluate(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('missing reconstruction key')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError(f'reconstruction shape must be {SHAPE}')
    if a.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError('reconstruction dtype must be float32 or float64')
    if not np.isfinite(a).all():raise ValueError('reconstruction contains nonfinite values')
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:r=z['reconstruction']
    if r.shape!=SHAPE or r.dtype not in [np.dtype('float32'),np.dtype('float64')] or not np.isfinite(r).all():raise ValueError('invalid evaluator reference')
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
    bounds={k:{'comparison':'>=' if k=='ncc' else '<=','boundary':config[k+'_boundary']} for k in ['ncc','nrmse','projection_residual']}
    metrics=compare(a,r)
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:d=z['measurements'][0];angles=z['angles'][0]
    matrix=ray_matrix(128,angles);scale=max(1.,float(np.max(abs(a))));err=0.
    for s in range(128):err+=float(np.sum((matrix.dot(a[0,s].astype(float).ravel()/scale)-d[:,:,s].ravel()/scale)**2))
    metrics['projection_residual']=float(min(1e300,np.sqrt(err)/np.linalg.norm(d)*min(scale,1e300)))
    passed=metrics['ncc']>=config['ncc_boundary'] and metrics['nrmse']<=config['nrmse_boundary'] and metrics['projection_residual']<=config['projection_residual_boundary']
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
