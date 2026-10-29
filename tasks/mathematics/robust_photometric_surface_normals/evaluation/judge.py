"""Standalone normal-field evaluator; independent of all solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent
def read_array(path,shape):
    with np.load(path,allow_pickle=False) as z:a=z['reconstruction']
    if a.shape!=tuple(shape) or a.dtype!=np.float64:raise ValueError('Invalid reconstruction shape/dtype')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a
def normal_metrics(image,reference,support):
    x=image[support];r=reference[support];a=x.ravel();b=r.ravel()
    nx=float(np.linalg.norm(a));nr=float(np.linalg.norm(b));dr=float(np.max(b)-np.min(b))
    if nr<=0 or dr<=0 or not np.isfinite([nx,nr,dr]).all():raise ValueError('Undefined reference or overflowing metric')
    ncc=float(a@b/(nx*nr)) if nx>0 else 0.
    nrmse=float(np.sqrt(np.mean((a-b)**2))/dr)
    xn=np.linalg.norm(x,axis=-1);rn=np.linalg.norm(r,axis=-1)
    if np.any(rn<=0):raise ValueError('Undefined reference orientation')
    # A zero submitted vector is an invalid orientation and incurs180degrees.
    angles=np.full(xn.shape,180.);valid=xn>0
    angles[valid]=np.degrees(np.arccos(np.clip(np.sum(x[valid]*r[valid],axis=-1)/(xn[valid]*rn[valid]),-1,1)))
    unit_error=float(np.max(abs(xn-1)))
    if (~support).any():unit_error=max(unit_error,float(np.max(abs(image[~support]))))
    result={'ncc':ncc,'nrmse':nrmse,'mean_angular_error_deg':float(np.mean(angles)),'unit_norm_max_error':unit_error}
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite metric')
    return result
def evaluate(prediction,asset_dir=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        e=Path(asset_dir);cfg=json.loads((e/'metrics.json').read_text(encoding='utf8'))
        s=cfg['outputs']['reconstruction']
        if set(cfg['outputs'])!={'reconstruction'} or s['dtype']!='float64' or len(s['shape'])!=4 or s['shape'][0]!=1 or s['shape'][-1]!=3:raise ValueError('Invalid evaluator output specification')
        thresholds={m:{'comparison':'>=' if m=='ncc' else '<=','boundary':float(cfg[m+'_boundary'])} for m in ['ncc','nrmse','mean_angular_error_deg','unit_norm_max_error']}
        b=[v['boundary'] for v in thresholds.values()]
        if not np.isfinite(b).all() or not(-1<=b[0]<=1) or any(v<=0 for v in b[1:]) or b[2]>180:raise ValueError('Invalid evaluator thresholds')
        ref=e/'ground_truth.npz'
        if not ref.exists():ref=e/'baseline_reference.npz'
        reference=read_array(ref,s['shape']);estimate=read_array(prediction,s['shape'])
        with np.load(e.parent/'data/raw_data.npz',allow_pickle=False) as z:support=z['support']
        if support.shape!=reference.shape[:-1] or support.dtype!=np.bool_ or not support.any():raise ValueError('Invalid evaluator public support')
        metrics=normal_metrics(estimate,reference,support)
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    result,code=evaluate(a.prediction)
    try:
        out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
