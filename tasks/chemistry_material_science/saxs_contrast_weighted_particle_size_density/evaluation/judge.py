"""Standalone evaluator: no solver imports or submitted code execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def load_arrays(path):
    if not Path(path).is_file():raise ValueError('File not found: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for key,shape in [('reconstruction',(1,500)),('background',(1,1))]:
            if key not in z:raise ValueError('Missing required key '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape for '+key)
            if a.dtype!=np.dtype('float64'):raise ValueError('Expected float64 for '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            if (a<0).any():raise ValueError('Negative '+key)
            arrays[key]=a
    return arrays
def evaluate(prediction,evaluation_dir=None):
    directory=HERE if evaluation_dir is None else Path(evaluation_dir)
    result=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        rules={'background_abs_error':'<=','observation_reduced_chi_squared':'<='}
        thresholds={k:dict(comparison=v,boundary=cfg[k+'_boundary']) for k,v in rules.items()}
        if not all(isinstance(v['boundary'],(int,float)) and np.isfinite(v['boundary']) for v in thresholds.values()):raise ValueError('Invalid evaluator boundary')
        pred=load_arrays(prediction);a=pred['reconstruction'].ravel()
        with np.load(directory.parent/'data/raw_data.npz',allow_pickle=False) as z:
            q=z['scattering_vector'][0];I=z['intensity'][0];error=z['intensity_uncertainty'][0]
        if any(v.shape!=(500,) or not np.isfinite(v).all() for v in [q,I,error]) or np.any(error<=0):raise ValueError('Invalid evaluator acquisition')
        expected_background=float(np.mean(I[(q>=.5)&(q<=1.)]))
        metrics=dict(background_abs_error=float(abs(pred['background'][0,0]-expected_background)))
        r=np.logspace(np.log10(10.),np.log10(30000.),500)
        widths=np.empty(500);widths[0]=r[1]-r[0];widths[-1]=r[-1]-r[-2];widths[1:-1]=(r[2:]-r[:-2])/2
        x=q[:,None]*r[None,:];F=np.empty_like(x);small=x<1e-3;t=x[small]**2;F[small]=1-t/10+t*t/280
        t=x[~small];F[~small]=3*(np.sin(t)-t*np.cos(t))/t**3
        K=(4./3.)*np.pi*r[None,:]**3*F**2*1e-4
        fitted=K@(a*widths)+float(pred['background'][0,0])
        metrics['observation_reduced_chi_squared']=float(np.mean(((fitted-I)/error)**2))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined metric')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
        result.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
        return result,0 if passed else 1
    except Exception as exc:
        result['errors']=[str(exc)];return result,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
