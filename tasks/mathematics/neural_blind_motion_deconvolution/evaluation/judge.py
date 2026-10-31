"""Standalone evaluator for a joint image/kernel submission; no solver imports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import fftconvolve

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,753,668),'kernel':(1,51,51)}


def read_prediction(path):
    if not Path(path).is_file():raise FileNotFoundError(f'prediction missing: {path}')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(SHAPES):raise ValueError('prediction requires exactly reconstruction and kernel')
        arrays={name:archive[name] for name in SHAPES}
    for key,value in arrays.items():
        if value.shape!=SHAPES[key]:raise ValueError(f'{key} shape must be {SHAPES[key]}')
        if value.dtype!=np.dtype('float32'):raise ValueError(f'{key} dtype must be float32')
        if not np.isfinite(value).all():raise ValueError(f'{key} contains NaN or infinity')
    return arrays


def score_arrays(image,kernel,truth,observation):
    """Declared integer gauge for image truth; original gauge for forward data."""
    x=np.asarray(image,dtype=np.float64);t=np.asarray(truth,dtype=np.float64)
    k=np.asarray(kernel,dtype=np.float64);y=np.asarray(observation,dtype=np.float64)
    corr=fftconvolve(t-t.mean(),(x-x.mean())[::-1,::-1],mode='full')
    middle=np.array(corr.shape)//2
    region=corr[middle[0]-25:middle[0]+26,middle[1]-25:middle[1]+26]
    offset=np.array(np.unravel_index(region.argmax(),region.shape))-25
    dy,dx=map(int,offset)
    target=t[50:-50,50:-50]
    estimate=x[50-dy:x.shape[0]-50-dy,50-dx:x.shape[1]-50-dx]
    denom=float(np.linalg.norm(target)*np.linalg.norm(estimate));scale=float(np.ptp(target))
    if scale<=0:raise ValueError('evaluator reference has zero dynamic range')
    ncc=0.0 if denom==0 else float(np.sum(estimate*target)/denom)
    predicted=fftconvolve(x,k[::-1,::-1],mode='valid')
    observed=y[25:-25,25:-25];observed_scale=float(np.ptp(observed))
    if observed_scale<=0:raise ValueError('evaluator observation has zero dynamic range')
    return {'ncc':ncc,'nrmse':float(np.sqrt(np.mean((estimate-target)**2))/scale),
            'forward_nrmse':float(np.sqrt(np.mean((predicted-observed)**2))/observed_scale),
            'kernel_mass_error':float(abs(k.sum()-1)),
            'kernel_negative_mass':float(np.maximum(-k,0).sum()),
            'image_min':float(x.min()),'image_max':float(x.max()),
            'shift_row':dy,'shift_column':dx}


def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        directions={'ncc':'>=','nrmse':'<=','forward_nrmse':'<=','kernel_mass_error':'<=','kernel_negative_mass':'<='}
        thresholds={name:{'comparison':relation,'boundary':float(config[name+'_boundary'])} for name,relation in directions.items()}
        thresholds.update(image_min={'comparison':'>=','boundary':0.0},image_max={'comparison':'<=','boundary':1.0})
        if not all(np.isfinite(v['boundary']) for v in thresholds.values()):raise ValueError('nonfinite evaluator threshold')
        report['thresholds']=thresholds
        arrays=read_prediction(prediction)
        with np.load(HERE/'ground_truth.npz',allow_pickle=False) as asset:truth=asset['reconstruction']
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as asset:observation=asset['observation']
        for label,value in [('reference',truth),('observation',observation)]:
            if value.shape!=(1,753,668) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError(f'invalid evaluator {label}')
        metrics=score_arrays(arrays['reconstruction'][0],arrays['kernel'][0],truth[0],observation[0])
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite evaluator metric')
        passed=all((metrics[name]>=v['boundary'] if v['comparison']=='>=' else metrics[name]<=v['boundary']) for name,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(Path(args.prediction))
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr);return 2
    return code


if __name__=='__main__':raise SystemExit(main())
