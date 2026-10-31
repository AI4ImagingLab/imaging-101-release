"""Independent exact-contract evaluator; hidden truth precedes method reference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def read(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'reconstruction','distortion'}:raise ValueError('Required keys: reconstruction,distortion')
        image=f['reconstruction'];a=f['distortion']
    if image.shape!=(1,506,688,3) or image.dtype!=np.uint8:raise ValueError('reconstruction must be uint8(1,506,688,3)')
    if a.shape!=(1,5) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('distortion must be finite float64(1,5)')
    return image,a
def score(image,a,reference,b):
    x=image.astype(float).ravel();y=reference.astype(float).ravel()
    span=float(np.ptp(y));yn=float(np.linalg.norm(y));xn=float(np.linalg.norm(x))
    if span==0 or yn==0 or np.any(b[0,[2,4]]==0):raise ValueError('Degenerate evaluator reference')
    return dict(ncc=float(np.dot(x,y)/(xn*yn)) if xn else 0.,nrmse=float(np.sqrt(np.mean((x-y)**2))/span),
                coefficient_relative=float(np.max(abs(a[0,[2,4]]/b[0,[2,4]]-1))),gauge_error=float(abs(a[0,0]-1)),odd_coefficient=float(np.max(abs(a[0,[1,3]]))))
def evaluate(prediction,assets=HERE):
    r=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        c=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        for name in ['ncc','nrmse','coefficient_relative','gauge_error','odd_coefficient']:
            v=float(c[name+'_boundary'])
            if not np.isfinite(v) or v<0:raise ValueError('Invalid threshold '+name)
            r['thresholds'][name]={'comparison':'>=' if name=='ncc' else '<=','boundary':v}
        ref=assets/'ground_truth.npz'
        if not ref.exists():ref=assets/'baseline_reference.npz'
        y,b=read(ref)
        if abs(b[0,0]-1)>1e-12 or np.any(abs(b[0,[1,3]])>1e-16):raise ValueError('Invalid evaluator reference gauge')
        x,a=read(prediction)
        with np.errstate(over='raise',invalid='raise',divide='raise'):m=score(x,a,y,b)
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Nonfinite score')
        passed=all(m[k]>=v['boundary'] if v['comparison']=='>=' else m[k]<=v['boundary'] for k,v in r['thresholds'].items())
        r.update(metrics=m,passed=passed,status='pass' if passed else 'fail');return r,0 if passed else 1
    except Exception as e:r['errors']=[str(e)];return r,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
