"""Strict spatial current-source NPZ evaluator; independent of reference solver."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
KEYS=('reconstruction',)
SHAPE=(1,17,17)
def compare(a,b):
    dynamic=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if not dynamic>0 or not norm>0:raise ValueError('degenerate hidden reference')
    scale=max(float(np.max(np.abs(a))),1.);u=a.ravel()/scale;v=b.ravel();un=float(np.linalg.norm(u))
    ncc=0. if un==0 else float(u@v/(un*norm))
    with np.errstate(over='ignore',invalid='ignore'):nrmse=float(np.sqrt(np.mean(((a-b)/dynamic)**2)))
    if not np.isfinite([ncc,nrmse]).all():raise ValueError('nonfinite metric arithmetic')
    return ncc,nrmse
def read_arrays(path):
    result={}
    with np.load(path,allow_pickle=False) as f:
        for key in KEYS:
            if key not in f:raise ValueError(f'missing key: {key}')
            a=f[key]
            if a.shape!=SHAPE:raise ValueError(f'{key}: expected shape {SHAPE}; got {a.shape}')
            if a.dtype!=np.dtype('float64'):raise ValueError(f'{key}: expected real float64')
            if not np.isfinite(a).all():raise ValueError(f'{key}: contains NaN or Infinity')
            result[key]=a
    return result
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        for key in KEYS:
            nb=float(config[key+'_ncc_boundary']);rb=float(config[key+'_nrmse_boundary'])
            if not np.isfinite([nb,rb]).all() or not 0<nb<=1 or not rb>0:raise ValueError('invalid metric boundaries')
            report['thresholds'][key+'_ncc']={'comparison':'>=','boundary':nb}
            report['thresholds'][key+'_nrmse']={'comparison':'<=','boundary':rb}
        ref=HERE/'ground_truth.npz'
        if not ref.exists():ref=HERE/'baseline_reference.npz'
        try:reference=read_arrays(ref)
        except Exception as e:raise ValueError(f'evaluator reference unavailable or invalid: {e}') from e
        value=read_arrays(Path(prediction));measured={}
        for key in KEYS:
            ncc,nrmse=compare(value[key],reference[key]);measured[key+'_ncc']=ncc;measured[key+'_nrmse']=nrmse
        passed=all(measured[k]>=t['boundary'] if t['comparison']=='>=' else measured[k]<=t['boundary'] for k,t in report['thresholds'].items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as e:report['errors']=[str(e)];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print(f'cannot write report: {e}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
