"""Standalone exact-contract SO2 profile judge, independent of solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def read_profile(path):
    if not Path(path).is_file():raise ValueError('Missing prediction/reference file')
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
        a=z['reconstruction']
    if a.shape!=(1,360):raise ValueError('Expected shape(1,360)')
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32/float64')
    if not np.isfinite(a).all():raise ValueError('Nonfinite profile')
    return a[0]
def score(a,b):
    span=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if span<=0 or norm<=0:raise ValueError('Undefined hidden-reference metrics')
    an=float(np.linalg.norm(a))
    ncc=0. if an==0 else float(np.dot(a,b)/(an*norm))
    nrmse=float(np.sqrt(np.mean((a-b)**2))/span)
    return {'ncc':ncc,'nrmse':nrmse}
def evaluate(path):
    r={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        a=read_profile(path);ref=HERE/'ground_truth.npz' if (HERE/'ground_truth.npz').is_file() else HERE/'baseline_reference.npz';b=read_profile(ref)
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));lo=float(cfg['ncc_boundary']);hi=float(cfg['nrmse_boundary'])
        if not np.isfinite([lo,hi]).all() or not 0<lo<=1 or hi<=0:raise ValueError('Invalid thresholds')
        m=score(a,b)
        if not np.isfinite(list(m.values())).all():raise ValueError('Nonfinite metrics')
        r['metrics']=m;r['thresholds']={'ncc':{'comparison':'>=','boundary':lo},'nrmse':{'comparison':'<=','boundary':hi}}
        ok=m['ncc']>=lo and m['nrmse']<=hi;r.update(status='pass' if ok else 'fail',passed=ok);return r,0 if ok else 1
    except Exception as exc:r.update(metrics={},errors=[str(exc)]);return r,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args();r,code=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
