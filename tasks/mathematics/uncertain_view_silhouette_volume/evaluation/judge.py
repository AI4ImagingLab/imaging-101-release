"""Standalone strict density/pose evaluator; no reference-solver import."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,128,128,128),'viewpoints':(1,8,3)}

def load(path):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(SHAPES):raise ValueError('Exactly reconstruction and viewpoints keys required')
        result={}
        for k,shape in SHAPES.items():
            a=z[k]
            if a.shape!=shape:raise ValueError(k+' shape must be '+str(shape))
            if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError(k+' must be real float32 or float64')
            if not np.isfinite(a).all():raise ValueError(k+' contains nonfinite values')
            result[k]=a.astype(np.float64)
        if np.min(result['reconstruction']) < -1e-6 or np.max(result['reconstruction'])>1+1e-6:raise ValueError('Density outside physical [0,1] float-roundoff allowance')
        return result

def rotations(angles):
    matrices=[]
    for x,y,z in angles:
        cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
        X=np.array([[1,0,0],[0,cx,sx],[0,-sx,cx]]);Y=np.array([[cy,0,-sy],[0,1,0],[sy,0,cy]]);Z=np.array([[cz,sz,0],[-sz,cz,0],[0,0,1]])
        matrices.append(Z@Y@X)
    return np.array(matrices)

def metrics(prediction,reference):
    x=np.asarray(prediction['reconstruction'],dtype=np.float64).ravel();y=np.asarray(reference['reconstruction'],dtype=np.float64).ravel();sx=np.max(np.abs(x));sy=np.max(np.abs(y));span=np.ptp(y)
    if sy==0 or span<=0:raise ValueError('Degenerate reference density')
    if sx==0:ncc=0.
    else:
        a=x/sx;b=y/sy
        ncc=float(np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))
    xp=x>.5;yp=y>.5;union=np.count_nonzero(xp|yp)
    if union==0:raise ValueError('Degenerate occupancy union')
    P=rotations(prediction['viewpoints'][0]);R=rotations(reference['viewpoints'][0])
    angle=np.arccos(np.clip((np.einsum('vij,vij->v',P,R)-1)/2,-1,1))
    return {'density_ncc':ncc,'density_nrmse':float(np.sqrt(np.mean((x-y)**2))/span),'occupancy_iou':float(np.count_nonzero(xp&yp)/union),'view_rmse_rad':float(np.sqrt(np.mean(angle**2))),'view_max_rad':float(np.max(angle))}

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'));bounds=config['thresholds']
        expected={'density_ncc':'>=','density_nrmse':'<=','occupancy_iou':'>=','view_rmse_rad':'<=','view_max_rad':'<='}
        if set(bounds)!=set(expected):raise ValueError('Invalid metric configuration')
        for k,op in expected.items():
            if bounds[k]['comparison']!=op or not np.isfinite(bounds[k]['boundary']):raise ValueError('Invalid threshold '+k)
            b=bounds[k]['boundary']
            if isinstance(b,bool) or b<=0 or (k in ('density_ncc','occupancy_iou') and b>1) or (k.startswith('view_') and b>np.pi):raise ValueError('Meaningless threshold '+k)
        report['thresholds']=bounds
        reference_path=ROOT/'ground_truth.npz'
        if not reference_path.is_file():reference_path=ROOT/'baseline_reference.npz'
        reference=load(reference_path);values=load(prediction);score=metrics(values,reference)
        passed=all((score[k]>=b['boundary'] if b['comparison']=='>=' else score[k]<=b['boundary']) for k,b in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=score)
        return report,0 if passed else 1
    except Exception as e:report['errors']=[str(e)];return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failure: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
