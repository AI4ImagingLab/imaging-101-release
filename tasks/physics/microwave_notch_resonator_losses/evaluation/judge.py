"""Independent strict all-output evaluator, no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CONTRACT={'resonance_frequency':((1,1),np.dtype('float64')),'quality_factors':((1,3),np.dtype('float64')),'reconstruction':((1,1001),np.dtype('complex128'))}

def validate(path):
    if not Path(path).is_file():raise ValueError(f'File unavailable: {path}')
    result={}
    with np.load(path,allow_pickle=False) as a:
        for key,(shape,dtype) in CONTRACT.items():
            if key not in a:raise ValueError(f'Missing key: {key}')
            arr=a[key]
            if arr.shape!=shape or arr.dtype!=dtype:raise ValueError(f'Invalid shape/dtype for {key}: {arr.shape}/{arr.dtype}')
            if not np.all(np.isfinite(arr)):raise ValueError(f'Nonfinite values in {key}')
            result[key]=arr
    return result

def score(pred,ref,observed):
    if np.any(ref['quality_factors']<=0):raise ValueError('Undefined nonpositive reference quality factor')
    def rel(a,b):
        norm=np.linalg.norm(b)
        if norm<=0:raise ValueError('Undefined reference norm')
        return float(np.linalg.norm(a-b)/norm)
    return {'frequency_relative_error':rel(pred['resonance_frequency'],ref['resonance_frequency']),
            'quality_factor_relative_error':float(np.max(np.abs(pred['quality_factors']/ref['quality_factors']-1))),
            'complex_response_relative_l2_error':rel(pred['reconstruction'],ref['reconstruction']),
            'observation_relative_rms':rel(pred['reconstruction'],observed)}

def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        pred=validate(path)
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=validate(reference)
        spec=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a: observed=a['scattering']
        metrics=score(pred,ref,observed)
        if any(not np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined nonfinite metric')
        thresholds={k:{'comparison':'<=','boundary':spec[k+'_boundary']} for k in metrics}
        if any(not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):raise ValueError('Invalid metric boundary')
        passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
