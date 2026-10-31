import argparse
import json
import sys
from pathlib import Path
import numpy as np


def positive_number(value):
    return type(value) in (int,float) and np.isfinite(value) and value>0


def forward(energy, membership, rates, times):
    # Stable logistic avoids exponential overflow without changing the model.
    z=-energy/(8.31446261815324*303.15)
    fraction=np.exp(-np.logaddexp(0.,-z))
    return .9*(membership@(-np.expm1(-(rates*fraction)[:,None]*times[None,:])))


def evaluate(prediction, root=None):
    root=Path(root) if root else Path(__file__).resolve().parent.parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((root/'evaluation/metrics.json').read_text(encoding='utf-8'))
        limits={k:cfg[k] for k in ['objective_boundary','forward_observation_normalized_l2_boundary']}
        if not all(positive_number(v) for v in limits.values()):raise ValueError('Invalid metric boundary')
        with np.load(root/'data/raw_data.npz',allow_pickle=False) as a:
            x=a['membership'][0];rates=a['intrinsic_rates'][0];times=a['time_s'][0]
            y=a['measurements'][0]/a['control'][0,:,None]*a['capacity'][0,:,None]
        if x.shape!=(63,146) or rates.shape!=(146,) or times.shape!=(7,) or y.shape!=(63,7):raise ValueError('Invalid evaluator observations')
        if not all(np.all(np.isfinite(v)) for v in [x,rates,times,y]) or np.linalg.norm(y)==0:raise ValueError('Invalid evaluator observation scale')
        truth=(root/'evaluation/ground_truth.npz').is_file()
        refpath=root/'evaluation/ground_truth.npz' if truth else root/'evaluation/baseline_reference.npz'
        def read(path):
            with np.load(path,allow_pickle=False) as a:
                out={}
                for key,shape in [('reconstruction',(1,146)),('uptake',(1,63,7))]:
                    if key not in a:raise ValueError('Missing output '+key)
                    v=a[key]
                    if v.shape!=shape:raise ValueError('Invalid shape '+key)
                    if v.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 '+key)
                    if not np.all(np.isfinite(v)):raise ValueError('Nonfinite '+key)
                    out[key]=v.copy()
                return out
        ref=read(refpath);out=read(prediction)
        refmodel=forward(ref['reconstruction'][0],x,rates,times)
        if np.linalg.norm(refmodel-ref['uptake'][0])/np.linalg.norm(y)>limits['forward_observation_normalized_l2_boundary']:raise ValueError('Inconsistent evaluator reference')
        expected=forward(out['reconstruction'][0],x,rates,times)
        with np.errstate(over='ignore',invalid='ignore'):
            obj=float(np.mean((expected-y)**2)+2e-4*np.mean(np.abs(np.diff(out['reconstruction'][0]))))
            consistency=float(np.linalg.norm(out['uptake'][0]-expected)/np.linalg.norm(y))
        report['metrics']={'objective':obj,'forward_observation_normalized_l2':consistency}
        report['thresholds']={k:dict(comparison='<=',boundary=float(limits[k+'_boundary'])) for k in report['metrics']}
        if truth:
            bound=cfg.get('truth_relative_l2_boundary')
            if not positive_number(bound):raise ValueError('Actual truth requires explicit truth_relative_l2_boundary')
            for key in out:
                scale=np.linalg.norm(ref[key])
                if scale==0:raise ValueError('Zero-norm truth requires explicit alternative metric')
                name='truth_relative_l2_'+key
                report['metrics'][name]=float(np.linalg.norm(out[key]-ref[key])/scale)
                report['thresholds'][name]=dict(comparison='<=',boundary=float(bound))
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite metric')
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
        return report,2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code


if __name__=='__main__':sys.exit(main())
