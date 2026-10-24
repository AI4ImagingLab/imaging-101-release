"""Independent standalone point/temporal-connectivity evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
NAMES=('point_recall','point_precision','position_rms_mm','edge_recall','edge_precision')
def read_tracks(path,truth=False):
    with np.load(Path(path),allow_pickle=False) as z:
        if not {'positions','frame_index','track_id'}<=set(z.files):raise ValueError('Missing required trajectory key')
        p,f,t=(z[k] for k in ['positions','frame_index','track_id'])
    if p.ndim!=3 or p.shape[0]!=1 or p.shape[2]!=3 or p.shape[1]<1:raise ValueError('positions must have shape(1,N,3),N>=1')
    n=p.shape[1]
    if p.dtype!=np.float64 or f.dtype!=np.int16 or t.dtype!=np.int32:raise ValueError('Required dtypes float64,int16,int32')
    if f.shape!=(1,n) or t.shape!=(1,n):raise ValueError('Mismatched trajectory row shapes')
    if not np.isfinite(p).all() or not np.isfinite(f).all() or not np.isfinite(t).all():raise ValueError('Nonfinite trajectory values')
    if np.any(f<0) or np.any(f>99) or np.any(t<0):raise ValueError('Invalid frame or segment identifier')
    # The source refiner is unconstrained after the prediction/triangulation
    # domain check. Preserve all resulting estimates, including slight overshoot;
    # outside points remain fully subject to precision/identity scoring.
    if truth and (np.any(p < -20) or np.any(p >20)):raise ValueError('Truth outside known acquisition volume')
    f=f[0];t=t[0];p=p[0]
    order=np.lexsort((f,t));a,b=order[:-1],order[1:]
    if np.any((t[a]==t[b])&(f[a]==f[b])):raise ValueError('Duplicate segment/frame pair')
    if truth and (n!=1250000 or not np.array_equal(np.unique(f),np.arange(100))):raise ValueError('Invalid full truth scope')
    return p,f,t
def read_limits(path):
    x=json.loads(Path(path).read_text(encoding='utf8'));limits={}
    for name in NAMES:
        v=x.get(name+'_boundary')
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError('Undefined metric boundary '+name)
        if name=='position_rms_mm':
            if v<=0 or v>=.1:raise ValueError('Position RMS boundary must be between0 and matching radius')
        elif not 0<v<=1:raise ValueError('Probability boundary must be in(0,1]')
        limits[name]=float(v)
    if x.get('matching_radius_mm')!=.1:raise ValueError('Fixed physical matching radius must be0.1mm')
    return limits
def measure(prediction,reference,return_details=False):
    p,f,t=prediction;q,g,u=reference
    matched=np.full(len(p),-1,np.int32);distances=np.full(len(p),np.inf);per_frame=[]
    for frame in range(100):
        truth=np.flatnonzero(g==frame);rows=np.flatnonzero(f==frame)
        if len(truth)==0:raise ValueError('Truth missing acquisition frame')
        used=set()
        if len(rows):
            distance,nn=cKDTree(q[truth]).query(p[rows]);order=np.argsort(distance,kind='stable')
            for j in order:
                if distance[j]>.1:break
                k=int(nn[j])
                if k in used:continue
                used.add(k);matched[rows[j]]=u[truth[k]];distances[rows[j]]=distance[j]
        per_frame.append([frame,len(rows),len(used),len(used)/len(truth),len(used)/len(rows) if len(rows) else 0.])
    good=np.isfinite(distances);count=int(good.sum())
    order=np.lexsort((f,t));a,b=order[:-1],order[1:];edges=(t[a]==t[b])&(f[b]==f[a]+1);a,b=a[edges],b[edges]
    correct=int(np.sum((matched[a]>=0)&(matched[a]==matched[b])))
    ix=np.lexsort((g,u));c,d=ix[:-1],ix[1:];true_edges=int(np.sum((u[c]==u[d])&(g[d]==g[c]+1)))
    if true_edges<=0:raise ValueError('No true temporal edges')
    values={'point_recall':count/len(q),'point_precision':count/len(p),'position_rms_mm':float(np.sqrt(np.mean(distances[good]**2))) if count else .1,'edge_recall':correct/true_edges,'edge_precision':correct/len(a) if len(a) else 0.}
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite measured metric')
    if return_details:return values,{'matched_identity':matched,'distance_mm':distances,'frame_metrics':np.asarray(per_frame),'predicted_edges':len(a),'correct_edges':correct,'true_edges':true_edges}
    return values
def evaluate(prediction_path,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        asset_dir=Path(asset_dir);limits=read_limits(asset_dir/'metrics.json')
        # Actual truth takes precedence. Invalid available truth cannot fall back.
        truth=asset_dir/'ground_truth.npz';ref=truth if truth.exists() else asset_dir/'baseline_reference.npz'
        reference=read_tracks(ref,truth=truth.exists());prediction=read_tracks(prediction_path)
        values=measure(prediction,reference);comparisons={k:('<=' if k=='position_rms_mm' else '>=') for k in NAMES}
        thresholds={k:{'comparison':comparisons[k],'boundary':limits[k]} for k in NAMES}
        passed=all(values[k]<=limits[k] if comparisons[k]=='<=' else values[k]>=limits[k] for k in NAMES)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=thresholds)
    except Exception as error:report['errors']=[type(error).__name__+': '+str(error)]
    return report
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True,type=Path);parser.add_argument('--report',required=True,type=Path);args=parser.parse_args(argv);report=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:print('Report-write failure: '+str(error),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
