"""Standalone evaluator; never imports submitted solver code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def load_arrays(path):
    if not Path(path).is_file():raise ValueError(f'Missing file: {path}')
    with np.load(path,allow_pickle=False) as data:
        result={}
        for key,shape in [('reconstruction',(1,7,10,2)),('confidence',(1,7,10))]:
            if key not in data:raise ValueError(f'Missing required key: {key}')
            a=data[key]
            if a.shape!=shape:raise ValueError(f'{key}: expected shape {shape}, got {a.shape}')
            if a.dtype != np.dtype('float64'):raise ValueError(f'{key}: expected real float64')
            if not np.isfinite(a).all():raise ValueError(f'{key}: nonfinite values')
            result[key]=a.astype(np.float64)
    return result

def score(pred,ref):
    a=pred['reconstruction'].ravel();b=ref['reconstruction'].ravel()
    norm=np.linalg.norm(a)*np.linalg.norm(b);span=np.ptp(b)
    cnorm=np.linalg.norm(ref['confidence'])
    if span<=0 or np.linalg.norm(b)==0 or cnorm==0:raise ValueError('Degenerate evaluator reference')
    return dict(ncc=float(np.dot(a,b)/norm) if norm else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/span),confidence_relative_l2=float(np.linalg.norm(pred['confidence']-ref['confidence'])/cnorm))

def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        root=Path(__file__).resolve().parent
        conf=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        thresholds={key:dict(comparison=op,boundary=conf[key+'_boundary']) for key,op in [('ncc','>='),('nrmse','<='),('confidence_relative_l2','<=')]}
        for t in thresholds.values():
            if not isinstance(t['boundary'],(int,float)) or not np.isfinite(t['boundary']):raise ValueError('Invalid metric boundary')
        reference=root/('ground_truth.npz' if (root/'ground_truth.npz').exists() else 'baseline_reference.npz')
        ref=load_arrays(reference);pred=load_arrays(prediction)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics=score(pred,ref)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite derived metrics: prediction scale is numerically invalid')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as e:
        report['errors']=[str(e)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:
        print(f'Cannot write report: {e}',file=sys.stderr);return 2
    print(json.dumps(report,allow_nan=False));return code

if __name__=='__main__':sys.exit(main())
