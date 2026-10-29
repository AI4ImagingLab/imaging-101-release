"""Independent surveyed-geometry and observation-consistency evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
NAMES={'microphone_ncc','microphone_nrmse','source_ncc','source_nrmse','timing_rms_seconds','odometry_rms_m'}

def cosine(estimate,reference):
    p=np.asarray(estimate).ravel();r=np.asarray(reference).ravel();pn=np.linalg.norm(p);rn=np.linalg.norm(r)
    if rn<=0:raise ValueError('Degenerate reference norm')
    return float(np.dot(p,r)/(pn*rn)) if pn else 0.

def range_nrmse(estimate,reference):
    span=np.ptp(reference)
    if span<=0:raise ValueError('Degenerate reference range')
    return float(np.sqrt(np.mean((estimate-reference)**2))/span)

def read_positions(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Require sole key reconstruction')
        p=z['reconstruction'].copy()
    if p.shape!=(1,24,3) or p.dtype!=np.float64:raise ValueError('Require real float64 shape(1,24,3)')
    if not np.isfinite(p).all():raise ValueError('Nonfinite geometry')
    return p[0]

def observed_residuals(positions,raw):
    """Profile linear clock nuisances; never use surveyed positions in this fit."""
    microphone=positions[:10];source=positions[10:]
    distance=np.linalg.norm(microphone[:,None,:]-source[None,:,:],axis=2)
    if np.any(distance<=0):raise ValueError('Coincident sensor/source has undefined timing geometry')
    dt=raw['emission_intervals'][0];time=np.cumsum(dt)
    design=np.zeros((256,19));rhs=np.empty(256)
    for i in range(10):
        for j in range(13):
            row=i*13+j;design[row,9+i]=dt[j+1]
            rhs[row]=raw['tdoa_successive'][0,j,i]-(distance[i,j+1]-distance[i,j])/340.-dt[j+1]
    for j in range(14):
        for i in range(1,10):
            row=130+9*j+i-1;design[row,i-1]=1;design[row,9+i]=time[j];design[row,9]=-time[j]
            rhs[row]=raw['tdoa_between'][0,i-1,j]-(distance[i,j]-distance[0,j])/340.
    clock=np.linalg.lstsq(design,rhs,rcond=None)[0]
    timing=float(np.sqrt(np.mean((design@clock-rhs)**2)))
    odometry=float(np.sqrt(np.mean((np.diff(source,axis=0)-raw['displacement'][0])**2)))
    return timing,odometry

def score(pred,truth,raw):
    result={}
    for name,indices in [('microphone',slice(0,10)),('source',slice(10,24))]:
        p=pred[indices].ravel();r=truth[indices].ravel();span=np.ptp(r);rn=np.linalg.norm(r);pn=np.linalg.norm(p)
        if span<=0 or rn<=0:raise ValueError('Degenerate surveyed '+name+' reference')
        result[name+'_ncc']=cosine(p,r)
        result[name+'_nrmse']=range_nrmse(p,r)
    result['timing_rms_seconds'],result['odometry_rms_m']=observed_residuals(pred,raw)
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Undefined evaluation metric')
    return result

def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        base=Path(asset_dir);config=json.loads((base/'metrics.json').read_text(encoding='utf8'));rules=config['thresholds']
        if set(rules)!=NAMES:raise ValueError('Wrong threshold names')
        for name,r in rules.items():
            expected='>=' if name.endswith('_ncc') else '<='
            if r['comparison']!=expected or not np.isfinite(r['boundary']) or r['boundary']<=0 or (name.endswith('_ncc') and r['boundary']>1):raise ValueError('Invalid threshold '+name)
        report['thresholds']=rules
        truth=read_positions(base/'ground_truth.npz');pred=read_positions(prediction)
        source=pred[10:]
        fixed=np.r_[source[0],source[1,1:],source[2,2]]
        if np.max(abs(fixed))>1e-9 or source[1,0]<=0 or source[2,1]<=0:raise ValueError('Output does not use declared first-three-event coordinate gauge')
        shapes={'tdoa_successive':(1,13,10),'tdoa_between':(1,9,14),'emission_intervals':(1,14),'displacement':(1,13,3)}
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in shapes}
        if any(a.shape!=shapes[k] or a.dtype!=np.float64 or not np.isfinite(a).all() for k,a in raw.items()):raise ValueError('Invalid evaluator observation assets')
        values=score(pred,truth,raw);passed=all(values[k]>=r['boundary'] if r['comparison']=='>=' else values[k]<=r['boundary'] for k,r in rules.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
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
