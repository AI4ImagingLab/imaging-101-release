"""Standalone reference agreement evaluator, independent of solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
KEYS={'reconstruction','fluorescence','innovations','dff','background_spatial','background_temporal','rigid_shifts','nonrigid_shifts'}
METRICS=('detection_f1','spatial_ncc','spatial_nrmse','fluorescence_ncc','fluorescence_nrmse','innovations_ncc','innovations_nrmse','dff_ncc','dff_nrmse','background_ncc','background_nrmse','motion_max_error')

def validated_thresholds(settings):
    thresholds={}
    for key in METRICS:
        name=key+'_boundary'
        boundary=settings.get(name)
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):
            raise ValueError(f'{name} boundary must be a finite number')
        bounded=key.endswith('_ncc') or key=='detection_f1'
        if boundary<0 or (bounded and boundary>1):
            raise ValueError(f'{name} boundary must be in [0,1]' if bounded else f'{name} boundary must be nonnegative')
        thresholds[key]={'comparison':'>=' if bounded else '<=','boundary':float(boundary)}
    return thresholds

def load_validated(path):
    if not Path(path).is_file():raise ValueError(f'File does not exist: {path}')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=KEYS:raise ValueError(f'Required exact keys: {sorted(KEYS)}')
        arrays={k:archive[k] for k in KEYS}
    a=arrays['reconstruction']
    if a.ndim!=4 or a.shape[0]!=1 or a.shape[2:]!=(170,170) or not 1<=a.shape[1]<=196:
        raise ValueError('reconstruction must have shape (1,K,170,170), 1<=K<=196')
    K=a.shape[1]
    shapes={'fluorescence':(1,K,3000),'innovations':(1,K,3000),'dff':(1,K,3000),'background_spatial':(1,2,170,170),'background_temporal':(1,2,3000),'rigid_shifts':(1,3000,2),'nonrigid_shifts':(1,3000,9,2)}
    for key,x in arrays.items():
        if key in shapes and x.shape!=shapes[key]:raise ValueError(f'{key} shape must be {shapes[key]}')
        if x.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(f'{key} dtype must be float32 or float64')
        if not np.isfinite(x).all():raise ValueError(f'{key} contains NaN or Infinity')
    for key in ['reconstruction','background_spatial']:
        x=arrays[key][0].reshape(arrays[key].shape[1],-1).astype(np.float64)
        if x.min() < -1e-7:raise ValueError(f'{key} must be nonnegative')
        if not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4,rtol=0):raise ValueError(f'{key} must have unit spatial L2 norm')
    if arrays['innovations'].min() < -1e-4:raise ValueError('innovations must be nonnegative within 1e-4 count float32 recurrence tolerance')
    return arrays

def pair_metrics(x,y):
    x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
    norm=float(np.linalg.norm(x)*np.linalg.norm(y));span=float(np.ptp(y))
    if np.linalg.norm(y)==0 or span==0:raise ValueError('Degenerate metric reference')
    return float(np.dot(x,y)/norm) if norm else 0.,float(np.sqrt(np.mean((x-y)**2))/span)

def score(prediction,reference):
    a=prediction['reconstruction'][0].reshape(prediction['reconstruction'].shape[1],-1).astype(float)
    b=reference['reconstruction'][0].reshape(reference['reconstruction'].shape[1],-1).astype(float)
    cosine=a@b.T/np.outer(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1))
    i,j=linear_sum_assignment(-cosine);good=cosine[i,j]>=.7;i=i[good];j=j[good]
    result={'detection_f1':float(2*len(i)/(len(a)+len(b)))}
    if not len(i):
        for label in ['spatial','fluorescence','innovations','dff']:result.update({label+'_ncc':0.,label+'_nrmse':1.})
    else:
        for key,label in [('reconstruction','spatial'),('fluorescence','fluorescence'),('innovations','innovations'),('dff','dff')]:
            ncc,error=pair_metrics(prediction[key][0,i],reference[key][0,j]);result[label+'_ncc']=ncc;result[label+'_nrmse']=error
    # Products remove scale/permutation and nonsingular rank-2 basis ambiguities.
    # Pixel blocks bound working memory; all observations are included.
    ps=prediction['background_spatial'][0].reshape(2,-1).T.astype(float);pt=prediction['background_temporal'][0].astype(float)
    rs=reference['background_spatial'][0].reshape(2,-1).T.astype(float);rt=reference['background_temporal'][0].astype(float)
    dot=nx=ny=error=0.;lo=np.inf;hi=-np.inf;count=0
    for start in range(0,len(ps),1024):
        x=ps[start:start+1024]@pt;y=rs[start:start+1024]@rt
        dot+=float(np.sum(x*y));nx+=float(np.sum(x*x));ny+=float(np.sum(y*y));error+=float(np.sum((x-y)**2));count+=x.size
        lo=min(lo,float(y.min()));hi=max(hi,float(y.max()))
    if ny<=0 or hi<=lo:raise ValueError('Degenerate background reference')
    result['background_ncc']=float(dot/np.sqrt(nx*ny)) if nx>0 else 0.;result['background_nrmse']=float(np.sqrt(error/count)/(hi-lo))
    result['motion_max_error']=max(float(np.max(abs(prediction[k]-reference[k]))) for k in ['rigid_shifts','nonrigid_shifts'])
    return result

def evaluate(prediction,assets=ROOT):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        settings=json.loads((Path(assets)/'metrics.json').read_text(encoding='utf-8'))
        thresholds=validated_thresholds(settings)
        reference=load_validated(Path(assets)/'baseline_reference.npz')
        pred=load_validated(prediction)
        result=score(pred,reference)
        if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metric')
        passed=all(v<=thresholds[k]['boundary'] if thresholds[k]['comparison']=='<=' else v>=thresholds[k]['boundary'] for k,v in result.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=result,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
