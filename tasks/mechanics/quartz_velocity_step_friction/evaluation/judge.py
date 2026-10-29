from pathlib import Path
import argparse,json,sys,numpy as np
SHAPES={'reconstruction':(1,3),'state':(1,914),'slider_velocity':(1,914),'friction':(1,914)}

def load(path):
    with np.load(path,allow_pickle=False) as f:
        arrays={}
        for k,shape in SHAPES.items():
            if k not in f:raise ValueError(f'Missing {k}')
            x=f[k]
            if x.shape!=shape:raise ValueError(f'{k} must have shape {shape}')
            if x.dtype!=np.float64:raise ValueError(f'{k} must have float64 dtype')
            if not np.all(np.isfinite(x)):raise ValueError(f'{k} contains NaN or Inf')
            arrays[k]=x
    return arrays

def score(pred,ref,observed,config):
    for role,values in [('prediction',pred),('reference',ref)]:
        if any(np.any(values[k]<=0) for k in ['reconstruction','state','slider_velocity']):raise ValueError('Nonpositive physical '+role)
    metrics={};thresholds={}
    for i,(name,boundary) in enumerate([('a_absolute_error','a_absolute_boundary'),('b_absolute_error','b_absolute_boundary'),('Dc_absolute_error_m','Dc_absolute_boundary_m')]):
        metrics[name]=float(abs(pred['reconstruction'][0,i]-ref['reconstruction'][0,i]));thresholds[name]={'comparison':'<=','boundary':config[boundary]}
    metrics['state_maximum_absolute_error_s']=float(np.max(abs(pred['state']-ref['state'])))
    thresholds['state_maximum_absolute_error_s']={'comparison':'<=','boundary':config['state_absolute_boundary_s']}
    norm=np.linalg.norm(ref['slider_velocity'])
    if norm<=0:raise ValueError('Undefined velocity reference normalization')
    metrics['velocity_relative_l2_error']=float(np.linalg.norm(pred['slider_velocity']-ref['slider_velocity'])/norm)
    thresholds['velocity_relative_l2_error']={'comparison':'<=','boundary':config['velocity_relative_l2_boundary']}
    metrics['friction_maximum_absolute_error']=float(np.max(abs(pred['friction']-ref['friction'])))
    thresholds['friction_maximum_absolute_error']={'comparison':'<=','boundary':config['friction_absolute_boundary']}
    metrics['observed_friction_RMS']=float(np.sqrt(np.mean((pred['friction'][0]-observed)**2)))
    thresholds['observed_friction_RMS']={'comparison':'<=','boundary':config['observed_friction_rms_boundary']}
    if any(not np.isfinite(t['boundary']) or t['boundary']<=0 for t in thresholds.values()):raise ValueError('Invalid metric threshold')
    if any(not np.isfinite(value) for value in metrics.values()):raise ValueError('Undefined or overflowing metric')
    return metrics,thresholds

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        base=Path(__file__).resolve().parent;truth=base/'ground_truth.npz';ref=load(truth if truth.exists() else base/'baseline_reference.npz')
        cfg=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as f:observed=f['friction'][0]
        metrics,thresholds=score(load(Path(args.prediction)),ref,observed,cfg);passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)]
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Report-write failure: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
