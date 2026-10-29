"""Standalone strict NPZ evaluator, independent of solver modules."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def score(a,b):
    a=a.ravel();b=b.ravel()
    sa=float(np.max(np.abs(a)));sb=float(np.max(np.abs(b)))
    span=float(np.ptp(b))
    if sb==0 or span==0:raise ValueError('Undefined metric: zero norm or constant reference')
    aa=a/sa if sa else np.zeros_like(a);bb=b/sb
    ncc=float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb))) if sa else 0.
    scale=max(sa,sb)
    rms=float(np.sqrt(np.mean((a/scale-b/scale)**2)))
    with np.errstate(over='ignore',invalid='ignore'):
        error=float(np.minimum(rms*(np.float64(scale)/span),1e300))
    return ncc,error
def observation_error(pred):
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:
        m=f['valid_mask'][0].astype(bool);s=f['measurements'][0][m].astype(np.float64)
    observed=np.abs(s[:,1]/s[:,0]);a=pred['reconstruction'][0][m]*np.pi/3
    predicted=(1+5*np.cos(a))/(5+np.cos(a))
    return float(np.sqrt(np.mean((predicted-observed)**2)))
def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((Path(asset_dir)/'metrics.json').read_text(encoding='utf-8'))
        contracts=config['outputs']
        reference_path=Path(asset_dir)/'ground_truth.npz'
        if not reference_path.exists(): reference_path=Path(asset_dir)/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as f: ref={k:f[k].copy() for k in contracts}
        with np.load(prediction,allow_pickle=False) as f: pred={k:f[k].copy() for k in contracts}
        for label,values in [('reference',ref),('prediction',pred)]:
            for k,c in contracts.items():
                a=values[k]
                if list(a.shape)!=c['shape']: raise ValueError(f'{label} {k}: shape {a.shape}, expected {c["shape"]}')
                if a.dtype!=np.dtype(c['dtype']): raise ValueError(f'{label} {k}: expected dtype {c["dtype"]}, got {a.dtype}')
                if not np.all(np.isfinite(a)): raise ValueError(f'{label} {k}: contains NaN or Inf')
        metrics={}; thresholds={}; passed=True
        for k in contracts:
            ncc,nrmse=score(pred[k],ref[k])
            metrics[k+'_ncc']=ncc;metrics[k+'_nrmse']=nrmse
            thresholds[k+'_ncc']={'comparison':'>=','boundary':config['ncc_boundary']}
            thresholds[k+'_nrmse']={'comparison':'<=','boundary':config['nrmse_boundary']}
            passed=passed and ncc>=config['ncc_boundary'] and nrmse<=config['nrmse_boundary']
        err=observation_error(pred)
        if not np.isfinite(err):err=1e300
        metrics['observation_nrmse']=err
        thresholds['observation_nrmse']={'comparison':'<=','boundary':config['observation_nrmse_boundary']}
        passed=passed and err<=config['observation_nrmse_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}']; return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(); report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
