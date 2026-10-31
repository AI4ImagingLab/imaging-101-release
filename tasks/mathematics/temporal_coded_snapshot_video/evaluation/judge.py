"""Standalone full-frame method agreement and all measured-exposure consistency."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,210,256,256)

def load_movie(path):
    if not Path(path).is_file():raise ValueError('Missing reconstruction asset or prediction: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Require sole key reconstruction')
        value=z['reconstruction']
    if value.shape!=SHAPE or value.dtype!=np.float64:raise ValueError('Require real float64 shape(1,210,256,256)')
    if not np.isfinite(value).all():raise ValueError('Nonfinite reconstruction')
    return value[0]

def frame_metrics(estimate,reference):
    """Uncentered cosine and dynamic-range NRMSE, independently known cases."""
    p=np.asarray(estimate).ravel();r=np.asarray(reference).ravel()
    span=np.ptp(r);rn=np.linalg.norm(r);pn=np.linalg.norm(p)
    if span<=0 or rn<=0:raise ValueError('Degenerate per-frame reference')
    cosine=float(np.dot(p,r)/(pn*rn)) if pn else 0.
    error=float(np.sqrt(np.mean((p-r)**2))/span)
    if not np.isfinite(cosine) or not np.isfinite(error):raise ValueError('Undefined or overflowing per-frame metric')
    return cosine,error

def score(movie,reference,observations,mask,config):
    values={};rules={}
    for burst,y in enumerate(observations):
        part=movie[burst*14:(burst+1)*14]
        if burst%2:part=part[::-1]
        prediction=np.zeros_like(y)
        for index in range(14):prediction+=mask[index]*part[index]
        tv=float(np.sum(abs(np.diff(part,axis=1)))+np.sum(abs(np.diff(part,axis=2))))
        values[f'anisotropic_tv_{burst}']=tv
        rules[f'anisotropic_tv_{burst}']={'comparison':'<=','boundary':config['anisotropic_tv_boundaries'][burst]}
        norm=np.linalg.norm(y)
        if norm<=0:raise ValueError('Degenerate observation norm')
        name=f'relative_observation_residual_{burst}'
        values[name]=float(np.linalg.norm(prediction-y)/norm)
        rules[name]={'comparison':'<=','boundary':config['relative_observation_residual_boundaries'][burst]}
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined observation metric')
    return values,rules

def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        base=Path(asset_dir);config=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        ob=config['relative_observation_residual_boundaries'];tv=config['anisotropic_tv_boundaries']
        if len(ob)!=15 or len(tv)!=15 or any(not np.isfinite(x) or x<=0 for x in ob+tv):raise ValueError('Invalid metric configuration')
        reference=None
        movie=load_movie(prediction)
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as z:
            if set(z.files)!={'mask','observations'}:raise ValueError('Invalid public acquisition assets')
            mask=z['mask'];y=z['observations']
        if mask.shape!=(1,14,256,256) or y.shape!=(1,15,256,256) or any(v.dtype!=np.float64 or not np.isfinite(v).all() for v in [mask,y]):raise ValueError('Invalid acquisition shape/dtype/value')
        values,rules=score(movie,reference,y[0],mask[0],config)
        passed=all(values[k]>=r['boundary'] if r['comparison']=='>=' else values[k]<=r['boundary'] for k,r in rules.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=rules)
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)])
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2

if __name__=='__main__':sys.exit(main())
