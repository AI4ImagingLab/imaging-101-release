"""Standalone normal-stress scorer, independent of all reconstruction code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent

def load_field(path):
    if not Path(path).is_file():raise ValueError('Missing prediction/reference NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Require exactly reconstruction')
        x=z['reconstruction']
    if x.shape!=(1,3166) or x.dtype!=np.float64:raise ValueError('Expected real float64 shape(1,3166)')
    if not np.isfinite(x).all():raise ValueError('Nonfinite normal stress')
    return x.ravel()

def field_metrics(x,y):
    sy=float(np.max(abs(y)));sx=float(np.max(abs(x)));span=float(np.ptp(y))
    if sy==0 or span<=0 or not np.isfinite(span):raise ValueError('Degenerate reference')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        ncc=0. if sx==0 else float(np.dot(x/sx,y/sy)/(np.linalg.norm(x/sx)*np.linalg.norm(y/sy)))
        nrmse=float(np.sqrt(np.mean(((x-y)/span)**2)))
        maximum=float(np.max(abs(x-y)))
    values=dict(ncc=ncc,nrmse=nrmse,max_abs_mpa=maximum)
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Metric overflow')
    return values

def evaluate(prediction,evaluation_dir=None):
    root=Path(evaluation_dir) if evaluation_dir is not None else HERE
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        keys=['ncc_boundary','nrmse_boundary','max_abs_boundary']
        for k in keys:
            v=cfg[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Invalid finite boundary '+k)
        if not -1<=cfg[keys[0]]<=1 or any(cfg[k]<0 for k in keys[1:]):raise ValueError('Invalid threshold range')
        path=root/'ground_truth.npz'
        if not path.exists():path=root/'baseline_reference.npz'
        reference=load_field(path);field_metrics(reference,reference)
    except Exception as error:report['errors']=['Evaluator error: '+str(error)];return report
    try:values=field_metrics(load_field(prediction),reference)
    except Exception as error:report['errors']=['Invalid prediction: '+str(error)];return report
    thresholds={name:dict(comparison=op,boundary=cfg[key]) for name,op,key in [('ncc','>=',keys[0]),('nrmse','<=',keys[1]),('max_abs_mpa','<=',keys[2])]}
    passed=all(v>=thresholds[k]['boundary'] if thresholds[k]['comparison']=='>=' else v<=thresholds[k]['boundary'] for k,v in values.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        out=Path(args.report);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:print('Report-write failure: '+str(error),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2

if __name__=='__main__':raise SystemExit(main())
