"""Independent standalone all-field Judge: actual truth has strict precedence."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
KEYS=('ktrans','extracellular_fraction')
SHAPE=(1,60,50)

def load_fields(path,role):
    if not Path(path).is_file():raise ValueError(f'{role} file does not exist: {path}')
    try:
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!=set(KEYS):
                raise ValueError(f'{role} requires exactly the keys {KEYS}')
            fields={k:archive[k] for k in KEYS}
    except Exception as exc:
        raise ValueError(f'Cannot load {role} NPZ: {exc}') from exc
    for key,a in fields.items():
        if a.shape!=SHAPE:raise ValueError(f'{role} {key} must have shape {SHAPE}')
        if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):
            raise ValueError(f'{role} {key} must be real float32 or float64')
        if not np.isfinite(a).all():raise ValueError(f'{role} {key} contains NaN or Infinity')
        cap=5.0 if key=='ktrans' else 1.0
        if np.any(a<0) or np.any(a>cap):
            raise ValueError(f'{role} {key} violates physical output bounds [0,{cap}]')
    return fields

def compare(estimate,reference):
    """Cosine NCC and reference dynamic-range-normalized RMS error."""
    x=estimate.ravel().astype(np.float64);y=reference.ravel().astype(np.float64)
    ynorm=float(np.linalg.norm(y));span=float(np.max(y)-np.min(y))
    if not math.isfinite(ynorm) or ynorm<=0 or not math.isfinite(span) or span<=0:
        raise ValueError('Reference has zero norm or zero dynamic range; metrics are undefined')
    xnorm=float(np.linalg.norm(x))
    ncc=0.0 if xnorm==0 else float(np.dot(x,y)/(xnorm*ynorm))
    nrmse=float(np.sqrt(np.mean((x-y)**2))/span)
    if not math.isfinite(ncc) or not math.isfinite(nrmse):
        raise ValueError('Metric calculation is nonfinite')
    return ncc,nrmse

def evaluate(prediction,asset_dir=ROOT):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    asset_dir=Path(asset_dir)
    try:
        with (asset_dir/'metrics.json').open(encoding='utf8') as f:config=json.load(f)
        quantities=config['quantities']
        if set(quantities)!=set(KEYS):raise ValueError('Metric configuration omits or adds a quantity')
        for key in KEYS:
            for metric,comparison in [('ncc','>='),('nrmse','<=')]:
                value=quantities[key][metric+'_boundary']
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('Metric boundary must be a finite real number')
                if (metric=='ncc' and not -1<=value<=1) or (metric=='nrmse' and value<0):
                    raise ValueError('Metric boundary is outside its meaningful range')
                report['thresholds'][key+'_'+metric]={'comparison':comparison,'boundary':value}
        truth=asset_dir/'ground_truth.npz'
        selected=truth if truth.exists() else asset_dir/'baseline_reference.npz'
        reference=load_fields(selected,'reference')
        # Validate the chosen reference before opening the submission.  A broken
        # actual truth must never trigger fallback to a method reference.
        for key in KEYS:compare(reference[key],reference[key])
    except Exception as exc:
        report['errors']=[f'Evaluator assets: {exc}'];return report,2
    try:fields=load_fields(prediction,'prediction')
    except Exception as exc:
        report['errors']=[str(exc)];return report,2
    try:
        measured={}
        for key in KEYS:
            ncc,nrmse=compare(fields[key],reference[key])
            measured[key+'_ncc']=ncc;measured[key+'_nrmse']=nrmse
        passed=all(value>=report['thresholds'][name]['boundary'] if name.endswith('_ncc')
                   else value<=report['thresholds'][name]['boundary'] for name,value in measured.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'Evaluator calculation: {exc}'];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write Judge report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
