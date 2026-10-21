"""Standalone evaluator: exact outputs, no implicit gauge fit or solver import."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent


def metrics(estimate,reference):
    """Uncentered cosine NCC, reference-dynamic-range RMS normalization."""
    r=reference.ravel();x=estimate.ravel();norm=np.linalg.norm(r);span=np.ptp(r)
    if norm<=0 or span<=0:raise ValueError('Degenerate evaluator reference')
    xnorm=np.linalg.norm(x)
    return float(np.dot(x,r)/(xnorm*norm)) if xnorm>0 else 0.,float(np.sqrt(np.mean((x-r)**2))/span)


def evaluate(prediction,asset_dir=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        specs=config['outputs']
        if not specs:raise ValueError('Empty required outputs')
        ncc=config['ncc_boundary'];nrmse=config['nrmse_boundary']
        if not np.isfinite(ncc) or not 0<ncc<=1 or not np.isfinite(nrmse) or nrmse<=0:raise ValueError('Invalid boundaries')
        path=assets/'ground_truth.npz'
        if not path.exists():path=assets/'baseline_reference.npz'
        with np.load(path,allow_pickle=False) as f:reference={k:f[k] for k in specs}
        with np.load(prediction,allow_pickle=False) as f:estimate={k:f[k] for k in specs}
        for label,arrays in [('prediction',estimate),('evaluator reference',reference)]:
            for key,spec in specs.items():
                a=arrays[key]
                if a.shape!=tuple(spec['shape']) or a.dtype!=np.dtype(spec['dtype']):raise ValueError('Invalid '+label+' shape/dtype: '+key)
                if not np.isfinite(a).all():raise ValueError('Nonfinite '+label+': '+key)
        values={};limits={}
        for key in specs:
            a,b=metrics(estimate[key],reference[key])
            values[key+'_ncc']=a;values[key+'_nrmse']=b
            limits[key+'_ncc']={'comparison':'>=','boundary':ncc};limits[key+'_nrmse']={'comparison':'<=','boundary':nrmse}
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined metric')
        passed=all(v>=limits[k]['boundary'] if limits[k]['comparison']=='>=' else v<=limits[k]['boundary'] for k,v in values.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=limits)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code


if __name__=='__main__':sys.exit(main())
