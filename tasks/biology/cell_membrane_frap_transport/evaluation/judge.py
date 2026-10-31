"""Standalone deterministic NPZ evaluator; no solver imports or execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,41,89),'diffusivity':(1,1)}

def load_arrays(path):
    if not Path(path).is_file(): raise ValueError(f'File does not exist: {path}')
    with np.load(path,allow_pickle=False) as archive:
        result={}
        for key,shape in SHAPES.items():
            if key not in archive: raise ValueError(f'Missing required key: {key}')
            value=archive[key]
            if value.shape!=shape: raise ValueError(f'{key}: expected shape {shape}, received {value.shape}')
            if value.dtype!=np.dtype('float64'): raise ValueError(f'{key}: expected real float64, received {value.dtype}')
            if not np.isfinite(value).all(): raise ValueError(f'{key}: non-finite values')
            result[key]=value
    return result

def metrics(prediction,reference):
    a=prediction['reconstruction'].ravel();b=reference['reconstruction'].ravel()
    span=np.ptp(b); bn=np.linalg.norm(b)
    if span<=0 or bn<=0 or reference['diffusivity'][0,0]<=0:
        raise ValueError('Degenerate hidden reference')
    an=np.linalg.norm(a)
    return {'ncc':float(a@b/(an*bn)) if an>0 else 0.,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/span),
            'diffusivity_relative_error':float(abs(prediction['diffusivity'][0,0]/reference['diffusivity'][0,0]-1))}

def evaluate(prediction_path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        thresholds={'ncc':{'comparison':'>=','boundary':cfg['ncc_boundary']},
                    'nrmse':{'comparison':'<=','boundary':cfg['nrmse_boundary']},
                    'diffusivity_relative_error':{'comparison':'<=','boundary':cfg['diffusivity_relative_error_boundary']}}
        if any(not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):
            raise ValueError('Invalid metric boundaries')
        report['thresholds']=thresholds
        refpath=HERE/'ground_truth.npz'
        if not refpath.exists(): refpath=HERE/'baseline_reference.npz'
        ref=load_arrays(refpath);pred=load_arrays(prediction_path)
        measured=metrics(pred,ref)
        if not all(np.isfinite(v) for v in measured.values()): raise ValueError('Non-finite evaluation metric')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        report['metrics']={}
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Report write failed: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
