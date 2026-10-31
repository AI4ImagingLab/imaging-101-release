"""Standalone deterministic NPZ evaluator, independent of submitted code."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

SHAPES={'reconstruction':(1,36,36,3),'mask':(1,36,36),'kernel':(1,169,137)}


def load_arrays(path):
    with np.load(path,allow_pickle=False) as data:
        arrays={}
        for key,shape in SHAPES.items():
            if key not in data:
                raise ValueError(f'Missing required key: {key}')
            value=data[key]
            if value.shape!=shape:
                raise ValueError(f'{key}: expected shape {shape}, received {value.shape}')
            if value.dtype!=np.dtype('float64'):
                raise ValueError(f'{key}: expected real float64, received {value.dtype}')
            if not np.isfinite(value).all():
                raise ValueError(f'{key}: nonfinite values')
            arrays[key]=value
    return arrays


def compare(estimate,reference):
    a,b=estimate.ravel(),reference.ravel()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    span=float(np.ptp(b))
    if not np.isfinite(denominator) or np.linalg.norm(b)==0 or span<=0:
        raise ValueError('Undefined metric: zero norm or constant comparison array.')
    return {'ncc':float(np.dot(a,b)/denominator) if denominator>0 else 0.0,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}


def model(arrays,background):
    h=arrays['kernel'][0]
    fields=[]
    for key in ('reconstruction','mask'):
        value=arrays[key][0]
        padded=np.zeros(background.shape if value.ndim==3 else background.shape[:2])
        padded[:36,:36,...]=value
        padded=np.roll(padded,(-18,-18),axis=(0,1))
        kernel=np.fft.fft2(h)
        if value.ndim==3:
            kernel=kernel[:,:,None]
        fields.append(np.fft.ifft2(kernel*np.fft.fft2(padded,axes=(0,1)),axes=(0,1)).real)
    return fields[0]+(1-fields[1][:,:,None])*background


def feasible_projection(arrays):
    result={k:v.copy() for k,v in arrays.items()}
    m=result['mask'][0].ravel();f=result['reconstruction'][0].reshape(-1,3)
    positive=np.maximum(f,0);ordered=np.sort(positive,axis=1)[:,::-1]
    cumulative=np.concatenate([np.zeros((m.size,1)),np.cumsum(ordered,axis=1)],axis=1)
    candidates=(m[:,None]+cumulative)/(1+np.arange(4))
    low=np.concatenate([ordered,np.zeros((m.size,1))],axis=1)
    high=np.concatenate([np.full((m.size,1),np.inf),ordered],axis=1)
    chosen=np.argmax((candidates>=low)&(candidates<=high),axis=1)
    opacity=np.clip(candidates[np.arange(m.size),chosen],0,1)
    result['mask']=opacity.reshape(1,36,36)
    result['reconstruction']=np.minimum(positive,opacity[:,None]).reshape(1,36,36,3)
    h=np.maximum(result['kernel'],0);total=h.sum()
    if total<=0:raise ValueError('Degenerate standard exposure mass')
    result['kernel']=h/total
    return result


def regularized_objective(arrays,observed,background):
    appearance=arrays['reconstruction'][0];opacity=arrays['mask'][0];kernel=arrays['kernel'][0]
    residual=model(arrays,background)-observed
    variation=0.
    for field in [appearance[...,i] for i in range(3)]+[opacity]:
        extended=np.pad(field,((1,1),(1,1)))
        dx=np.diff(extended,axis=1)[1:,:]
        dy=np.diff(extended,axis=0)[:,1:]
        variation+=float(np.hypot(dx,dy).sum())
    yy,xx=np.indices((36,36));x=xx+1-18.5;y=yy+1-18.5
    rotated=np.zeros((36,36));count=np.zeros((36,36))
    for angle in np.deg2rad([16.,25.,78.,152.]):
        rx=np.rint(x*np.cos(angle)+y*np.sin(angle)+18.5).astype(int)-1
        ry=np.rint(-x*np.sin(angle)+y*np.cos(angle)+18.5).astype(int)-1
        keep=(rx>=0)&(rx<36)&(ry>=0)&(ry<36)
        rotated[keep]+=opacity[ry[keep],rx[keep]];count[keep]+=1
    np.divide(rotated,count,out=rotated,where=count>0)
    symmetry=float(np.sum((rotated-opacity)**2))
    value=.5*float(np.sum(residual**2))+2**(-12)*variation+.005*symmetry
    violation=max(float(np.maximum(-kernel,0).max()),abs(float(kernel.sum())-1),float(np.maximum(-opacity,0).max()),float(np.maximum(opacity-1,0).max()),float(np.maximum(-appearance,0).max()),float(np.maximum(appearance-opacity[...,None],0).max()))
    return value,violation


def evaluate(prediction,assets=None):
    root=Path(assets) if assets is not None else Path(__file__).resolve().parent
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        estimate=load_arrays(prediction)
        reference_path=root/'ground_truth.npz'
        if not reference_path.is_file():
            reference_path=root/'baseline_reference.npz'
        reference=load_arrays(reference_path)
        limits=json.loads((root/'metrics.json').read_text(encoding='utf-8'))
        for key in ['regularized_objective_ratio_boundary','constraint_violation_boundary','observation_rmse_boundary']:
            if isinstance(limits[key],bool) or not np.isfinite(limits[key]) or limits[key]<=0:raise ValueError('Invalid metric boundary: '+key)
        measured={};thresholds={}
        with np.load(root.parent/'data/raw_data.npz',allow_pickle=False) as raw:
            observed,background=raw['observation'][0],raw['background'][0]
        if observed.shape!=(169,137,3) or background.shape!=observed.shape or not np.isfinite(observed).all() or not np.isfinite(background).all():
            raise ValueError('Invalid evaluator calibration data.')
        candidate,violation=regularized_objective(estimate,observed,background)
        attainable,_=regularized_objective(feasible_projection(reference),observed,background)
        if attainable<=0 or not np.isfinite(attainable):raise ValueError('Degenerate standard objective')
        measured['regularized_objective_ratio']=candidate/attainable
        measured['constraint_violation']=violation
        thresholds['regularized_objective_ratio']={'comparison':'<=','boundary':float(limits['regularized_objective_ratio_boundary'])}
        thresholds['constraint_violation']={'comparison':'<=','boundary':float(limits['constraint_violation_boundary'])}
        measured['observation_rmse']=float(np.sqrt(np.mean((model(estimate,background)-observed)**2)))
        thresholds['observation_rmse']={'comparison':'<=','boundary':float(limits['observation_rmse_boundary'])}
        for value in measured.values():
            if not np.isfinite(value):
                raise ValueError('Nonfinite evaluation result.')
        for threshold in thresholds.values():
            if not np.isfinite(threshold['boundary']):
                raise ValueError('Nonfinite boundary.')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds)
    except Exception as exc:
        report['errors'].append(str(exc))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr)
        return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]


if __name__=='__main__':
    raise SystemExit(main())
