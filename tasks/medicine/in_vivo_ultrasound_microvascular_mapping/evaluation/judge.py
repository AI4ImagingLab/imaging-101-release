"""Standalone submission evaluator. Never imports or runs reference solver."""
import argparse,json,sys
from pathlib import Path
import numpy as np

SHAPE=(1,831,1281)
KEYS=('reconstruction','speed')

def evaluate(prediction,asset_dir=None):
    root=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        for key in KEYS:
            for metric,comparison in [('ncc','>= '),('nrmse','<= ')]:
                name=f'{key}_{metric}';value=config[name+'_boundary']
                if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value):raise ValueError('Invalid metric boundary')
                if metric=='ncc' and not -1<=value<=1:raise ValueError('NCC boundary outside [-1,1]')
                if metric=='nrmse' and value<0:raise ValueError('Negative NRMSE boundary')
                report['thresholds'][name]={'comparison':comparison.strip(),'boundary':value}
        refs=root/'ground_truth.npz'
        if not refs.exists():refs=root/'baseline_reference.npz'
        with np.load(refs,allow_pickle=False) as f:
            reference={k:f[k] for k in KEYS}
        for k,a in reference.items():
            if a.shape!=SHAPE or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid hidden reference '+k)
            if np.any(a<0):raise ValueError('Negative hidden physical reference '+k)
            if np.ptp(a)==0 or np.linalg.norm(a)==0:raise ValueError('Degenerate hidden reference '+k)
        with np.load(Path(prediction),allow_pickle=False) as f:
            estimate={k:f[k] for k in KEYS}
        for k,a in estimate.items():
            if a.shape!=SHAPE:raise ValueError(f'{k}: required shape {SHAPE}')
            if a.dtype!=np.float64:raise ValueError(f'{k}: required float64')
            if not np.isfinite(a).all():raise ValueError(k+': nonfinite values')
            if np.any(a<0):raise ValueError(k+': negative physical output')
        measured={}
        for k in KEYS:
            a=estimate[k].ravel();b=reference[k].ravel()
            sa=float(np.max(np.abs(a)));sb=float(np.max(np.abs(b)))
            aa=a/sa if sa>0 else a;bb=b/sb
            norm=np.linalg.norm(aa)*np.linalg.norm(bb)
            measured[k+'_ncc']=float(np.dot(aa,bb)/norm) if norm>0 else 0.0
            error=np.abs(a-b);scale=float(np.max(error))
            rms=scale*float(np.sqrt(np.mean((error/scale)**2))) if scale>0 else 0.0
            measured[k+'_nrmse']=rms/float(np.ptp(b))
            if not np.isfinite(measured[k+'_nrmse']):raise ValueError('Metric exceeds finite numeric range')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in report['thresholds'].items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as error:
        report['errors']=[str(error)];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as error:
        print('Cannot write report: '+str(error),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
