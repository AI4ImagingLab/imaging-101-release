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
        ncc=float(limits['ncc_boundary']);nrmse=float(limits['nrmse_boundary'])
        if not(-1<=ncc<=1 and np.isfinite(nrmse) and nrmse>0):
            raise ValueError('Invalid metric boundaries.')
        measured={};thresholds={}
        for key in SHAPES:
            for metric,value in compare(estimate[key],reference[key]).items():
                name=f'{key}_{metric}'
                measured[name]=value
                thresholds[name]={'comparison':'>=' if metric=='ncc' else '<=',
                                  'boundary':ncc if metric=='ncc' else nrmse}
        with np.load(root.parent/'data/raw_data.npz',allow_pickle=False) as raw:
            observed,background=raw['observation'][0],raw['background'][0]
        if observed.shape!=(169,137,3) or background.shape!=observed.shape or not np.isfinite(observed).all() or not np.isfinite(background).all():
            raise ValueError('Invalid evaluator calibration data.')
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
