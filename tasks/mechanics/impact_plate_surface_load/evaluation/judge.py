"""Standalone pressure and independent load-cell evaluation; no solver imports."""
from pathlib import Path
import argparse,json,sys,numpy as np
ROOT=Path(__file__).resolve().parent
SHAPE=(1,250,60,78)

def ncc_nrmse(estimate,reference):
    a=estimate.ravel();b=reference.ravel();den=float(np.linalg.norm(a)*np.linalg.norm(b));span=float(np.ptp(b))
    if den<=0 or span<=0:raise ValueError('undefined NCC/NRMSE for degenerate arrays')
    return float(np.dot(a,b)/den),float(np.sqrt(np.mean((a-b)**2))/span)

def evaluate(prediction: str | Path) -> tuple[dict,int]:
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        with np.load(prediction,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('missing required key reconstruction')
            a=f['reconstruction']
        if a.shape!=SHAPE:raise ValueError('required shape '+str(SHAPE))
        if a.dtype!=np.float64:raise ValueError('required real float64 dtype')
        if not np.isfinite(a).all():raise ValueError('nonfinite prediction')
        refpath=ROOT/'ground_truth.npz'
        if not refpath.exists():refpath=ROOT/'baseline_reference.npz'
        with np.load(refpath,allow_pickle=False) as f:b=f['reconstruction']
        if b.shape!=SHAPE or b.dtype!=np.float64 or not np.isfinite(b).all():raise ValueError('invalid hidden image reference')
        with np.load(ROOT/'baseline_reference.npz',allow_pickle=False) as f:time=f['measurement_time'];force=f['measurement_force']
        if time.ndim!=1 or force.shape!=time.shape or not np.isfinite(time).all() or not np.isfinite(force).all() or np.any(np.diff(time)<=0):raise ValueError('invalid hidden independent force measurements')
        cfg=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        for key in ['ncc_boundary','nrmse_boundary','force_nrmse_boundary']:
            value=cfg[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('invalid finite numeric threshold '+key)
            if key=='ncc_boundary' and not -1<=value<=1:raise ValueError('NCC threshold outside [-1,1]')
            if key!='ncc_boundary' and value<0:raise ValueError('negative error threshold '+key)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            nc,nr=ncc_nrmse(a,b);target=np.interp(np.arange(250)/75000.,time,force)
            if np.ptp(target)<=0:raise ValueError('constant force measurement')
            integrated=np.sum(a[0,:,20:50,20:50],axis=(1,2))*0.0030065919701586503**2
            fn=float(np.sqrt(np.mean((integrated-target)**2))/np.ptp(target))
        metrics={'ncc':nc,'nrmse':nr,'force_nrmse':fn};thresholds={'ncc':{'comparison':'>=','boundary':cfg['ncc_boundary']},'nrmse':{'comparison':'<=','boundary':cfg['nrmse_boundary']},'force_nrmse':{'comparison':'<=','boundary':cfg['force_nrmse_boundary']}}
        if not all(np.isfinite(x) for x in metrics.values()):raise ValueError('undefined metric')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items());report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds);return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args();r,c=evaluate(a.prediction)
    try:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return c

if __name__=='__main__':raise SystemExit(main())
