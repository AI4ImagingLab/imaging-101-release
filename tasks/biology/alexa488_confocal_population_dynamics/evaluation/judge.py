import argparse,json,sys
from pathlib import Path
import numpy as np

def positive_number(value):
    return type(value) in (int,float) and np.isfinite(value) and value>0


def evaluate(prediction, assets=None):
    base=Path(assets) if assets else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((base/'metrics.json').read_text())
        reference_path=base/'ground_truth.npz' if (base/'ground_truth.npz').is_file() else base/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as r:
            refs={k:r[k].copy() for k in ['reconstruction','correlation']}
        for k,s in [('reconstruction',(1,3)),('correlation',(1,200))]:
            if refs[k].shape!=s or refs[k].dtype!=np.dtype('float64') or not np.all(np.isfinite(refs[k])):raise ValueError('Invalid evaluator reference '+k)
        refp=refs['reconstruction'][0]
        if not (0<refp[0]<=1e5 and 0<refp[1]<=.95 and .002<=refp[2]<=1e3):raise ValueError('Nonphysical evaluator reference')
        if np.any(refs['reconstruction']==0) or np.linalg.norm(refs['correlation'])==0:raise ValueError('Degenerate evaluator reference')
        for key in ['parameter_relative_max','curve_relative_l2','model_consistency_relative_l2','observation_sse']:
            bound=config[key+'_boundary']
            if not positive_number(bound):raise ValueError('Invalid evaluator boundary '+key)
        with np.load(prediction,allow_pickle=False) as a:
            out={}
            for k,s in [('reconstruction',(1,3)),('correlation',(1,200))]:
                if k not in a:raise ValueError('Missing key '+k)
                x=a[k]
                if x.shape!=s:raise ValueError('Wrong shape '+k)
                if x.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 '+k)
                if not np.all(np.isfinite(x)):raise ValueError('Nonfinite '+k)
                out[k]=x.copy()
        p=out['reconstruction'][0]
        if not (0<p[0]<=1e5 and 0<=p[1]<=.95 and .002<=p[2]<=1e3):raise ValueError('Nonphysical parameter values')
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as d:
            t=d['lag_ms'][0];g=d['correlation'][0]
        expected=(1+p[1]/(1-p[1])*np.exp(-t/.002))/(p[0]*(1+t/p[2])*np.sqrt(1+t/(25*p[2])))
        report['metrics']={
            'parameter_relative_max':float(np.max(np.abs((p-refs['reconstruction'][0])/refs['reconstruction'][0]))),
            'curve_relative_l2':float(np.linalg.norm(out['correlation']-refs['correlation'])/np.linalg.norm(refs['correlation'])),
            'model_consistency_relative_l2':float(np.linalg.norm(out['correlation'][0]-expected)/np.linalg.norm(expected)),
            'observation_sse':float(np.sum((out['correlation'][0]-g)**2))}
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite measured metric')
        report['thresholds']={k:dict(comparison='<=',boundary=config[k+'_boundary']) for k in report['metrics']}
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(metrics={},errors=[str(exc)])
        return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
