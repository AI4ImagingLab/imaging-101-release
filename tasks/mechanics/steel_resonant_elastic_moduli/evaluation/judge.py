"""Independent strict evaluator for isotropic moduli and complete spectra."""
from pathlib import Path
import argparse,json,sys
import numpy as np
SHAPES={'reconstruction':(1,2),'resonances':(1,76)}

def load(path):
    if not path.is_file():raise ValueError(f'Missing file: {path}')
    with np.load(path,allow_pickle=False) as f:
        result={}
        for key,shape in SHAPES.items():
            if key not in f:raise ValueError(f'Missing required key: {key}')
            a=f[key]
            if a.shape!=shape:raise ValueError(f'{key} must have shape {shape}')
            if a.dtype!=np.float64:raise ValueError(f'{key} must be float64')
            if not np.all(np.isfinite(a)):raise ValueError(f'{key} contains nonfinite numbers')
            result[key]=a
    return result

def pair(observed,calculated):
    """Independent shortest monotone path, permitting unobserved resonances."""
    n,m=len(observed),len(calculated)
    if np.any(calculated<=0) or np.any(np.diff(calculated)<0):raise ValueError('Predicted resonances must be positive and sorted')
    previous=np.zeros(m+1);paths=[[] for _ in range(m+1)]
    for i,value in enumerate(observed):
        current=np.full(m+1,np.inf);newpaths=[[] for _ in range(m+1)]
        for j in range(i+1,m+1):
            candidate=previous[j-1]+(value/calculated[j-1]-1)**2
            if candidate<=current[j-1]:current[j]=candidate;newpaths[j]=paths[j-1]+[j-1]
            else:current[j]=current[j-1];newpaths[j]=newpaths[j-1]
        previous,paths=current,newpaths
    return np.asarray(paths[-1],dtype=np.int64)

def score(pred,ref,config,observation,weights):
    metrics={};thresholds={}
    for i,name in enumerate(['c11_absolute_error_GPa','c44_absolute_error_GPa']):
        metrics[name]=float(abs(pred['reconstruction'][0,i]-ref['reconstruction'][0,i]));thresholds[name]={'comparison':'<=','boundary':config['moduli_absolute_boundary_GPa']}
    metrics['maximum_frequency_error_MHz']=float(np.max(abs(pred['resonances']-ref['resonances'])))
    thresholds['maximum_frequency_error_MHz']={'comparison':'<=','boundary':config['frequency_absolute_boundary_MHz']}
    theoretical=pred['resonances'][0];indices=pair(observation,theoretical)
    metrics['observation_rms_percent']=float(100*np.sqrt(np.sum(weights*(observation/theoretical[indices]-1)**2)/np.count_nonzero(weights)))
    thresholds['observation_rms_percent']={'comparison':'<=','boundary':config['observation_rms_percent_boundary']}
    for name,entry in thresholds.items():
        if not np.isfinite(entry['boundary']) or entry['boundary']<=0:raise ValueError(f'Invalid threshold {name}')
    return metrics,thresholds

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        base=Path(__file__).resolve().parent
        cfg=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        truth=base/'ground_truth.npz';reference=load(truth if truth.exists() else base/'baseline_reference.npz')
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as f:observations,weights=f['frequencies'][0],f['analysis_weights'][0]
        pred=load(Path(args.prediction));metrics,thresholds=score(pred,reference,cfg,observations,weights)
        passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)]
    try:
        out=Path(args.report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Report-write failure: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
