"""Independent empirical-depth evaluator; never import solver/submitted code."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
TRUTH=HERE/'ground_truth.npz';REFERENCE=HERE/'baseline_reference.npz';METRICS=HERE/'metrics.json';RAW=HERE.parent/'data/raw_data.npz'
SHAPE=(1,3,650,650)

def pair_metrics(estimate,reference):
    x=np.asarray(estimate,dtype=np.float64).ravel();y=np.asarray(reference,dtype=np.float64).ravel()
    scale=float(np.ptp(y));ny=float(np.linalg.norm(y));nx=float(np.linalg.norm(x))
    if scale<=0 or ny<=0:raise ValueError('Degenerate reference has no defined range/NCC')
    ncc=float(np.dot(x,y)/nx/ny) if nx else 0.
    return {'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(np.sqrt(np.mean((x-y)**2))/scale),'maximum_absolute_error_mm':float(np.max(np.abs(x-y)))}

def _observation_residual(raw,estimate,mask,frame):
    """Independent nearest calibrated axial depth, then float64 RGB residual.

    No reference geometry enters this forward check. The index is uniquely
    specified by the nearest supplied physical axial depth, earliest at ties.
    """
    rows=np.flatnonzero(mask.ravel());squared=[]
    lc=raw['left_color'][0].reshape(3,-1,25).astype(np.float64);rc=raw['right_color'][0].astype(np.float64)
    ld=raw['left_depth'][0].reshape(-1,25).astype(np.float64);rd=raw['right_depth'][0].astype(np.float64)
    p=raw['pattern'][0,frame].reshape(-1,3);w=raw['white'][0,frame].reshape(-1,3);target=estimate.ravel()
    for start in range(0,len(rows),2048):
        r=rows[start:start+2048];depth=ld[r]@rd
        index=np.abs(depth-target[r,None]).argmin(axis=1)
        predicted=np.stack([np.sum(lc[c,r]*rc[c,:,index],axis=1) for c in range(3)],axis=1)
        observed=(p[r]/w[r]).astype(np.float32).astype(np.float64)
        squared.append(np.sum((predicted-observed)**2,axis=1))
    return float(np.sqrt(np.mean(np.concatenate(squared))))

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads(METRICS.read_text(encoding='utf8'))
        limits={k:config[k+'_boundary'] for k in ['ncc','nrmse','maximum_absolute_error_mm']}
        for k,v in limits.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or (not -1<=v<=1 if k=='ncc' else v<0):raise ValueError('Invalid boundary '+k)
        obs=config['observation_rms_boundaries']
        if len(obs)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in obs):raise ValueError('Invalid observation boundaries')
        reference_path=TRUTH if TRUTH.exists() else REFERENCE
        with np.load(reference_path,allow_pickle=False) as f:reference=f['reconstruction']
        if reference.shape!=SHAPE or reference.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(reference).all():raise ValueError('Invalid hidden reference shape/dtype/finiteness')
    except Exception as e:
        report['errors']=['Evaluator error: '+str(e)];return report,2
    try:
        with np.load(Path(prediction),allow_pickle=False) as f:
            if set(f.files)!={'reconstruction'}:raise ValueError('Required sole key reconstruction')
            estimate=f['reconstruction']
        if estimate.shape!=SHAPE:raise ValueError('Expected batch/frame/row/column shape'+str(SHAPE))
        if estimate.dtype!=np.dtype('float32'):raise ValueError('Expected real float32 millimetres')
        if not np.isfinite(estimate).all():raise ValueError('Nonfinite estimate')
    except Exception as e:
        report['errors']=['Invalid prediction: '+str(e)];return report,2
    try:
        with np.load(RAW,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        shapes={'pattern':(1,3,650,650,3),'white':(1,3,650,650,3),'left_color':(1,3,650,650,25),'right_color':(1,3,25,749),'left_depth':(1,650,650,25),'right_depth':(1,25,749)}
        if set(raw)!=set(shapes):raise ValueError('Invalid public keys')
        for k,shape in shapes.items():
            dtype=np.dtype('uint16' if k in ('pattern','white') else 'float32')
            if raw[k].shape!=shape or raw[k].dtype!=dtype or not np.isfinite(raw[k]).all():raise ValueError('Invalid public array '+k)
        minimum=raw['white'][0].min(axis=-1);mask=minimum>.05*minimum.max(axis=(-2,-1),keepdims=True)
        if mask.shape!=(3,650,650) or np.any(mask.sum(axis=(-2,-1))==0):raise ValueError('Empty support')
        for i in range(3):
            if np.any(reference[0,i][~mask[i]]!=-1):raise ValueError('Invalid hidden sentinel')
            pair_metrics(reference[0,i][mask[i]],reference[0,i][mask[i]])
    except Exception as e:
        report['errors']=['Evaluator error: '+str(e)];return report,2
    if any(np.any(estimate[0,i][~mask[i]]!=-1) for i in range(3)):
        report['errors']=['Invalid prediction: unsupported depths must be exactly-1mm'];return report,2
    values={};thresholds={}
    try:
        for i in range(3):
            result=pair_metrics(estimate[0,i][mask[i]],reference[0,i][mask[i]])
            for k,value in result.items():
                name=f'frame{i}_{k}';values[name]=value;thresholds[name]={'comparison':'>=' if k=='ncc' else '<=','boundary':limits[k]}
            name=f'frame{i}_observation_rms';values[name]=_observation_residual(raw,estimate[0,i],mask[i],i);thresholds[name]={'comparison':'<=','boundary':obs[i]}
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Metric arithmetic is nonfinite')
    except Exception as e:
        report['errors']=['Evaluator scoring error: '+str(e)];return report,2
    passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in thresholds.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
    return report,0 if passed else 1

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        output=Path(a.report);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
