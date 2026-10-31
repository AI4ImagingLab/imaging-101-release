"""Independent standalone NPZ evaluator; no reference-solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.interpolate import PchipInterpolator,BSpline
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,100),'bin_weights':(1,77),'cumulative_weights':(1,100)}
def load_arrays(path):
    if not Path(path).is_file():raise ValueError('File not found: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape '+key)
            if a.dtype!=np.dtype('float64'):raise ValueError('Expectedfloat64 '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            if np.any(a<0):raise ValueError('Negative '+key)
            result[key]=a
    return result
def evaluate(prediction,evaluation_dir=None):
    directory=HERE if evaluation_dir is None else Path(evaluation_dir)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        rules={'weights_ncc':'>=','weights_nrmse':'<=','density_ncc':'>=','density_nrmse':'<=','cumulative_max_abs_error':'<=','observation_relative_l2':'<=','spline_consistency_max_abs':'<='}
        thresholds={k:dict(comparison=v,boundary=cfg[k+'_boundary']) for k,v in rules.items()}
        if not all(isinstance(v['boundary'],(int,float)) and np.isfinite(v['boundary']) for v in thresholds.values()):raise ValueError('Invalid metricboundary')
        refpath=directory/'ground_truth.npz'
        if not refpath.exists():refpath=directory/'baseline_reference.npz'
        ref=load_arrays(refpath);pred=load_arrays(prediction);m={}
        for name,key in [('weights','bin_weights'),('density','reconstruction')]:
            a=pred[key].ravel();b=ref[key].ravel();norm=np.linalg.norm(a)*np.linalg.norm(b)
            if np.linalg.norm(b)==0 or np.ptp(b)<=0:raise ValueError('Degenerate evaluator reference')
            m[name+'_ncc']=float(a@b/norm) if norm>0 else 0.
            m[name+'_nrmse']=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
        m['cumulative_max_abs_error']=float(np.max(abs(pred['cumulative_weights']-ref['cumulative_weights'])))
        with np.load(directory.parent/'data/raw_data.npz',allow_pickle=False) as z:
            q=z['relative_pressure'][0];y=z['loading'][0];p=z['kernel_relative_pressure'][0];cal=z['kernel_loading'][0];w=z['pore_width'][0]
        if q.shape!=(96,) or y.shape!=(96,) or w.shape!=(77,) or cal.shape!=(len(p),77):raise ValueError('Malformed evaluator calibration')
        if not all(np.isfinite(a).all() for a in [q,y,p,cal,w]):raise ValueError('Nonfinite evaluator calibration')
        matrix=PchipInterpolator(np.r_[0.,p],np.vstack([np.zeros(77),cal]),axis=0,extrapolate=False)(q)
        model=matrix@pred['bin_weights'][0]
        if np.linalg.norm(y)==0:raise ValueError('Empty evaluator observationnorm')
        m['observation_relative_l2']=float(np.linalg.norm(model-y)/np.linalg.norm(y))
        knots=np.r_[np.zeros(2),np.arange(76),np.full(2,75)];u=np.linspace(0,75,100)
        axis=BSpline(knots,w,2)(u);density=BSpline(knots,pred['bin_weights'][0]/np.ediff1d(w,to_begin=w[0]),2)(u)
        cumulative=np.cumsum(density*np.ediff1d(axis,to_begin=axis[0]))
        m['spline_consistency_max_abs']=float(max(np.max(abs(pred['reconstruction'][0]-density)),np.max(abs(pred['cumulative_weights'][0]-cumulative))))
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Nonfinite metric')
        passed=all(m[k]>=v['boundary'] if v['comparison']=='>=' else m[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=m,thresholds=thresholds);return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-writefailure: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
