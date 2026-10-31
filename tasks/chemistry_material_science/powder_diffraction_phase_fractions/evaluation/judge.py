"""Standalone mixed-provenance Judge: weighed composition truth, native profile/shift reference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,8,7),'fitted':(1,8,3001),'shifts':(1,8)}
def load(path,shapes):
    with np.load(path,allow_pickle=False) as f:
        out={}
        for key,shape in shapes.items():
            if key not in f: raise ValueError('Missing key '+key)
            a=f[key]
            if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all(): raise ValueError('Invalid shape, float32/64 dtype, or finite values: '+key)
            out[key]=a.astype(float)
    return out
def interpolate(x,y,target):
    cs=CubicSpline(x,y,bc_type='natural',axis=0); clipped=np.clip(target,x[0],x[-1]); delta=target-clipped
    return cs(clipped)+(delta[:,None] if y.ndim==2 else delta)*cs(clipped,1)
def evaluate(path):
    cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
    estimate=load(path,SHAPES); truth=load(HERE/'ground_truth.npz',{'reconstruction':(1,8,7)})['reconstruction'][0]
    ref=load(HERE/'baseline_reference.npz',SHAPES)
    x=estimate['reconstruction'][0]; profile=estimate['fitted'][0]; shifts=estimate['shifts'][0]
    if np.any(x<0) or np.any(x>100) or np.any(profile<0): raise ValueError('Negative profile or composition outside0..100wt%')
    if np.any(abs(x[:,0]-20)>1e-4): raise ValueError('Known corundum fraction must be20wt%')
    if np.any(abs(shifts)>.3): raise ValueError('Angular shift outside declared range')
    if np.any(truth<0) or not np.allclose(truth.sum(axis=1),100): raise ValueError('Invalid weighed truth')
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:
        angles=f['angles'][0]; counts=f['counts'][0]; standards=f['standards'][0]; rir=f['rir'][0]
    metrics={}; thresholds={}
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for i in range(8):
            if np.ptp(ref['fitted'][0,i])<=0: raise ValueError('Degenerate profile reference')
            m={'composition_rmse':float(np.sqrt(np.mean((x[i,1:]-truth[i,1:])**2))),
               'observed_rwp':float(np.sqrt(np.sum((profile[i]-counts[i])**2/counts[i])/np.sum(counts[i]))),
               'fitted_nrmse':float(np.sqrt(np.mean((profile[i]-ref['fitted'][0,i])**2))/np.ptp(ref['fitted'][0,i])),
               'shift_absolute':float(abs(shifts[i]-ref['shifts'][0,i]))}
            step=float(np.mean(np.diff(angles))); lower=max(angles[0]+shifts[i],angles[0]); upper=min(angles[-1]+shifts[i],angles[-1])
            grid=lower+np.arange(int(np.floor((upper-lower)/step+1e-10))+1)*step
            relative=x[i]*rir/(20*rir[0])
            model=interpolate(grid-shifts[i],interpolate(angles,standards,grid)@relative,angles)
            scale=float(model@profile[i]/(model@model))
            if scale<=0 or np.linalg.norm(profile[i])==0: raise ValueError('Nonpositive overall intensity scale')
            m['fraction_scale_consistency']=float(np.linalg.norm(profile[i]-scale*model)/np.linalg.norm(profile[i]))
            for key,value in m.items():
                name=f'mix{i+1}_{key}'; bound=cfg[key+'_boundary'][i] if key in ['composition_rmse','observed_rwp'] else cfg[key+'_boundary']
                if not np.isfinite(value) or not np.isfinite(bound) or bound<=0: raise ValueError('Nonfinite metric or invalid boundary')
                metrics[name]=value; thresholds[name]={'comparison':'<=','boundary':float(bound)}
    passed=all(value<=thresholds[key]['boundary'] for key,value in metrics.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':metrics,'thresholds':thresholds,'errors':[]}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prediction',required=True); parser.add_argument('--report',required=True); args=parser.parse_args()
    try: report=evaluate(Path(args.prediction)); code=0 if report['passed'] else 1
    except Exception as exc: report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc) or type(exc).__name__]}; code=2
    try:
        path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc: print('Report write failed: '+str(exc),file=sys.stderr); return 2
    return code
if __name__=='__main__': sys.exit(main())
