"""Strict standalone four-field evaluator, independent of any solver."""
from pathlib import Path
import json,argparse,sys
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPES={'strain':(1,3,40,40),'lattice_spacing':(1,40,40)}

def load_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError(f'Missing {key}')
            a=z[key]
            if a.shape!=shape or a.dtype!=np.float64:raise ValueError(f'Require real float64 {key}{shape}')
            if not np.isfinite(a).all():raise ValueError(f'Nonfinite {key}')
            result[key]=a
    return result

def pair_metrics(x,reference):
    a=x.ravel();b=reference.ravel();norm=np.linalg.norm(b);span=np.ptp(b)
    if norm<=0 or span<=0:raise ValueError('Degenerate evaluator reference')
    na=np.linalg.norm(a)
    return float(a@b/(na*norm)) if na>0 else 0.,float(np.sqrt(np.mean((a-b)**2))/span)

def score(prediction,reference):
    out={}
    pairs=[('strain_xx',prediction['strain'][0,0],reference['strain'][0,0]),('strain_xy',prediction['strain'][0,1],reference['strain'][0,1]),('strain_yy',prediction['strain'][0,2],reference['strain'][0,2]),('lattice_spacing',prediction['lattice_spacing'],reference['lattice_spacing'])]
    for name,x,r in pairs:out[name+'_ncc'],out[name+'_nrmse']=pair_metrics(x,r)
    return out

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'));bounds=cfg['boundaries']
        reference=load_prediction(ASSETS/'ground_truth.npz' if (ASSETS/'ground_truth.npz').exists() else ASSETS/'baseline_reference.npz');x=load_prediction(prediction)
        with np.errstate(over='raise',invalid='raise',divide='raise'):m=score(x,reference)
        if set(bounds)!=set(m):raise ValueError('Missing or unexpected evaluator boundaries')
        thresholds={};ok=True
        for key,val in m.items():
            bound=bounds[key]
            if isinstance(bound,bool) or not isinstance(bound,(int,float)) or not np.isfinite(bound) or not 0<bound<=1:raise ValueError(f'Invalid boundary {key}')
            if not np.isfinite(val):raise ValueError(f'Nonfinite metric {key}')
            greater=key.endswith('_ncc');thresholds[key]={'comparison':'>=' if greater else '<=','boundary':bound};ok=ok and (val>=bound if greater else val<=bound)
        report.update(status='pass' if ok else 'fail',passed=bool(ok),metrics=m,thresholds=thresholds);return report,0 if ok else 1
    except Exception as e:report['errors']=[f'{type(e).__name__}: {e}'];return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:print(f'Cannot write report: {e}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
