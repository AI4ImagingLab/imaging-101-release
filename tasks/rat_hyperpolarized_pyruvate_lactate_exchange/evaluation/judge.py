"""Standalone strict two-field NPZ evaluator; never imports solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
FIELDS=('exchange_rate','initial_lactate')
def read_fields(path):
    if not Path(path).is_file():raise ValueError('missing NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(FIELDS):raise ValueError('exactly two required field keys expected')
        out={k:z[k] for k in FIELDS}
    for k,a in out.items():
        if a.shape!=(1,32,32):raise ValueError(k+': expected (1,32,32)')
        if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(k+': expected real float32/float64')
        if not np.isfinite(a).all():raise ValueError(k+': nonfinite values')
    return out
def limits(directory):
    m=json.loads((Path(directory)/'metrics.json').read_text(encoding='utf8'))
    for k in ('ncc_boundary','nrmse_boundary','max_abs_over_range_boundary','negative_count_boundary'):
        v=m.get(k)
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError('invalid metric boundary '+k)
    if not 0<m['ncc_boundary']<=1:raise ValueError('invalid NCC boundary')
    if m['nrmse_boundary']<0 or m['max_abs_over_range_boundary']<0 or m['negative_count_boundary']!=0:raise ValueError('invalid error/nonnegative boundary')
    return m
def evaluate(prediction,directory=HERE):
    result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        directory=Path(directory);m=limits(directory)
        refpath=directory/'ground_truth.npz'
        if not refpath.exists():refpath=directory/'baseline_reference.npz'
        ref=read_fields(refpath);pred=read_fields(prediction)
        metrics={};thresholds={}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for k in FIELDS:
                a=pred[k].astype(np.float64).ravel();b=ref[k].astype(np.float64).ravel();dr=float(np.ptp(b));norm=float(np.linalg.norm(b))
                if dr<=0 or norm<=0 or np.any(b<0):raise ValueError('invalid nondegenerate nonnegative reference '+k)
                an=float(np.linalg.norm(a));ncc=float(a@b/(an*norm)) if an>0 else 0.
                values={'ncc':ncc,'dr_nrmse':float(np.sqrt(np.mean((a-b)**2))/dr),'max_abs_over_range':float(abs(a-b).max()/dr),'negative_count':int(np.count_nonzero(a<0))}
                cfg={'ncc':('>=',m['ncc_boundary']),'dr_nrmse':('<=',m['nrmse_boundary']),'max_abs_over_range':('<=',m['max_abs_over_range_boundary']),'negative_count':('<=',m['negative_count_boundary'])}
                for n,v in values.items():
                    if not np.isfinite(v):raise ValueError('nonfinite measured metric '+k+'/'+n)
                    metrics[k+'/'+n]=v;op,boundary=cfg[n];thresholds[k+'/'+n]={'comparison':op,'boundary':boundary}
        passed=all(v>=thresholds[k]['boundary'] if thresholds[k]['comparison']=='>=' else v<=thresholds[k]['boundary'] for k,v in metrics.items())
        result.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as e:result['errors']=[type(e).__name__+': '+str(e)]
    return result
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv);r=evaluate(a.prediction)
    try:
        report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:print('Cannot write report: '+str(e),file=sys.stderr);return 2
    return 0 if r['status']=='pass' else 1 if r['status']=='fail' else 2
if __name__=='__main__':sys.exit(main())
