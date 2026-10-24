"""Standalone APb-FRET Judge. No reference-solver or submitted-code imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,512,512),'correction_factors':(1,4)}

def _load(path):
    if not Path(path).is_file():raise ValueError('NPZ file does not exist: '+str(path))
    with np.load(path,allow_pickle=False) as a:
        if set(a.files)!=set(SHAPES):raise ValueError('require exactly reconstruction and correction_factors')
        out={k:a[k] for k in SHAPES}
    for k,x in out.items():
        if x.shape!=SHAPES[k]:raise ValueError(k+': wrong batch-first shape')
        if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(k+': require real float32 or float64')
        if not np.isfinite(x).all():raise ValueError(k+': nonfinite values')
    return out

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=['ncc_boundary','nrmse_boundary','max_abs_boundary','factor_max_abs_boundary']
        for k in names:
            x=cfg[k]
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x):raise ValueError('undefined threshold '+k)
        if not -1<=cfg['ncc_boundary']<=1 or any(cfg[k]<0 for k in names[1:]):raise ValueError('invalid threshold range')
        # Actual truth always takes precedence; an invalid truth is an evaluator error.
        truth=HERE/'ground_truth.npz'
        reference=_load(truth if truth.exists() else HERE/'baseline_reference.npz')
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:mask=a['observation_mask']
        if mask.shape!=(1,512,512) or mask.dtype!=np.bool_ or not mask.any():raise ValueError('invalid public support')
        if np.any(reference['reconstruction'][~mask]!=0):raise ValueError('reference padding must be zero')
        y=reference['reconstruction'][mask].astype(np.float64)
        if np.linalg.norm(y)==0 or np.ptp(y)==0:raise ValueError('degenerate efficiency reference')
    except Exception as exc:
        report['errors']=['Evaluator error: '+str(exc)];return report
    report['thresholds']={'efficiency_ncc':{'comparison':'>=','boundary':cfg['ncc_boundary']},'efficiency_nrmse':{'comparison':'<=','boundary':cfg['nrmse_boundary']},'efficiency_max_abs':{'comparison':'<=','boundary':cfg['max_abs_boundary']},'factor_max_abs':{'comparison':'<=','boundary':cfg['factor_max_abs_boundary']}}
    try:
        out=_load(prediction)
        if np.any(out['reconstruction'][~mask]!=0):raise ValueError('support-exterior efficiency must be exactly zero')
        x=out['reconstruction'][mask].astype(np.float64)
        norm=float(np.linalg.norm(x));ncc=float(x@y/norm/np.linalg.norm(y)) if norm else 0.
        metrics={'efficiency_ncc':ncc,'efficiency_nrmse':float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)),'efficiency_max_abs':float(np.max(abs(x-y))),'factor_max_abs':float(np.max(abs(out['correction_factors'].astype(np.float64)-reference['correction_factors'].astype(np.float64))))}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('metric overflow')
    except Exception as exc:
        report['errors']=['Invalid prediction: '+str(exc)];return report
    passed=all(v>=report['thresholds'][k]['boundary'] if report['thresholds'][k]['comparison']=='>=' else v<=report['thresholds'][k]['boundary'] for k,v in metrics.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    return report

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2

if __name__=='__main__':raise SystemExit(main())
