"""Standalone dual-output evaluator; independent of all solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy import signal,stats
HERE=Path(__file__).resolve().parent
SHAPES={'hrf':(1,400,37),'neural_drive':(1,400,152)}
def load_arrays(path):
    if not Path(path).is_file():raise ValueError('file not found: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES):raise ValueError('require exact keys hrf,neural_drive')
        out={k:f[k].copy() for k in SHAPES}
    for k,a in out.items():
        if a.shape!=SHAPES[k]:raise ValueError('shape mismatch '+k)
        if a.dtype!=np.dtype('float64'):raise ValueError('required float64 '+k)
        if not np.isfinite(a).all():raise ValueError('nonfinite '+k)
    return out
def cosine(estimate,reference):
    a=estimate.ravel();b=reference.ravel();den=np.linalg.norm(a)*np.linalg.norm(b)
    if not np.linalg.norm(b):raise ValueError('degenerate reference norm')
    return float(np.dot(a,b)/den) if den else 0.
def nrmse(estimate,reference):
    span=np.ptp(reference)
    if not span:raise ValueError('degenerate reference range')
    return float(np.sqrt(np.mean((estimate-reference)**2))/span)
def filtered_observations(bold):
    x=stats.zscore(np.ascontiguousarray(bold[0].T),axis=0,ddof=1)
    # Independent literal Fourier filtering, including the original first pass.
    for low,high in [(0.,sys.float_info.max),(.01,.08)]:
        mean=x.mean(axis=0,keepdims=True);extended=np.concatenate((x,x[::-1]),axis=0)
        spec=np.fft.fft(extended,axis=0);index=np.arange(len(extended));freq=np.minimum(index,len(extended)-index)/(2*len(extended))
        spec[(freq<low)|(freq>=high)]=0
        x=np.fft.ifft(spec,axis=0).real[:len(x)]+mean
    return x
def observation_residual(pred,observed):
    drive=pred['neural_drive'][0].T;h=signal.resample_poly(pred['hrf'][0].T,1,3,axis=0)
    H=np.fft.fft(np.pad(h,((0,len(drive)-len(h)),(0,0))),axis=0)
    modeled=np.fft.ifft(H*np.fft.fft(drive,axis=0),axis=0).real
    denom=np.sqrt(np.mean(observed**2,axis=0))
    if np.any(denom<=0):raise ValueError('degenerate observed ROI')
    return float(np.max(np.sqrt(np.mean((modeled-observed)**2,axis=0))/denom))
def score(pred,ref,observed):
    metrics={}
    for k in SHAPES:
        metrics[k+'_ncc']=cosine(pred[k],ref[k]);metrics[k+'_nrmse']=nrmse(pred[k],ref[k])
        spans=np.ptp(ref[k][0],axis=1)
        if np.any(spans<=0):raise ValueError('constant reference ROI')
        metric='worst_roi_hrf_nrmse' if k=='hrf' else 'worst_roi_drive_nrmse'
        metrics[metric]=float(np.max(np.sqrt(np.mean((pred[k][0]-ref[k][0])**2,axis=1))/spans))
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined or overflowing evaluation metric')
    metrics['worst_roi_observation_residual']=observation_residual(pred,observed)
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return metrics
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        bounds=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))['boundaries']
        names={'hrf_ncc','hrf_nrmse','neural_drive_ncc','neural_drive_nrmse','worst_roi_hrf_nrmse','worst_roi_drive_nrmse','worst_roi_observation_residual'}
        if set(bounds)!=names or any(not np.isfinite(v) or v<=0 for v in bounds.values()):raise ValueError('invalid evaluator boundaries')
        report['thresholds']={k:{'comparison':'>=' if k.endswith('_ncc') else '<=','boundary':v} for k,v in bounds.items()}
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load_arrays(reference);pred=load_arrays(path)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:observed=filtered_observations(f['bold'])
        values=score(pred,ref,observed)
        passed=all(values[k]>=v if k.endswith('_ncc') else values[k]<=v for k,v in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args();report,code=evaluate(a.prediction)
    try:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
