"""Standalone evaluator: explicit declared arrays, no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def relative_error(estimate,reference):
    norm=float(np.linalg.norm(reference.ravel()))
    if not np.isfinite(norm) or norm<=0:raise ValueError('Evaluator reference has zero or invalid norm')
    value=float(np.linalg.norm((estimate-reference).ravel())/norm)
    if not np.isfinite(value):raise ValueError('Nonfinite relative error')
    return value
def read_arrays(path,schema,label):
    if not Path(path).is_file():raise ValueError(f'{label} file does not exist: {path}')
    with np.load(path,allow_pickle=False) as f:
        result={}
        for key,spec in schema.items():
            if key not in f:raise ValueError(f'{label} missing required key {key}')
            v=f[key]
            if list(v.shape)!=spec['shape']:raise ValueError(f'{label} {key} shape {v.shape}, expected {spec["shape"]}')
            if str(v.dtype)!=spec['dtype']:raise ValueError(f'{label} {key} dtype {v.dtype}, expected {spec["dtype"]}')
            if not np.isfinite(v).all():raise ValueError(f'{label} {key} contains NaN or infinity')
            result[key]=v
    return result
def physical_metrics(arrays,raw_path):
    with np.load(raw_path,allow_pickle=False) as z:
        w=z['angular_frequency'][0];storage=z['storage_modulus'][0];loss=z['loss_modulus'][0]
    order=np.argsort(w);w=w[order];omega=np.geomspace(w.min(),w.max(),100)
    data=np.r_[np.interp(omega,w,storage[order]),np.interp(omega,w,loss[order])]
    tau=np.geomspace(np.exp(-np.pi/2)/w.max(),np.exp(np.pi/2)/w.min(),100)
    delta=np.diff(np.log(tau));weights=np.r_[delta[0]/2,(delta[:-1]+delta[1:])/2,delta[-1]/2]
    h=arrays['reconstruction'][0];g,t=arrays['maxwell_modes'][0]
    if np.any(h<=0) or np.any(g<=0) or np.any(t<=0) or np.any(np.diff(t)<=0):raise ValueError('Positive spectrum/strengths and strictly ascending positive times required')
    if np.any(t<.02/w.max()) or np.any(t>50/w.min()):raise ValueError('Mode times outside declared physical model support')
    def matrix(times):
        x=omega[:,None]*times[None,:];return np.vstack([x*x/(1+x*x),x/(1+x*x)])
    fits=np.stack([matrix(tau)@(weights*h),matrix(t)@g])
    return {'continuous_relative_sse':float(np.sum((fits[0]/data-1)**2)),'discrete_relative_sse':float(np.sum((fits[1]/data-1)**2)),'log_spectrum_curvature':float(np.sum(np.diff(np.log(h),n=2)**2)),'forward_identity_relative':float(np.linalg.norm(arrays['fitted_moduli'][0]-fits)/np.linalg.norm(fits))}


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        schema=config['output_schema']
        expected={'reconstruction':[1,100],'maxwell_modes':[1,2,13],'fitted_moduli':[1,2,200]}
        if set(schema)!=set(expected) or any(schema[k].get('shape')!=v or schema[k].get('dtype')!='float64' for k,v in expected.items()):raise ValueError('Invalid evaluator output coverage/schema')
        names=['continuous_relative_sse','discrete_relative_sse','log_spectrum_curvature','forward_identity_relative']
        thresholds={k:{'comparison':'<=','boundary':config[k+'_boundary']} for k in names}
        if any(not isinstance(v['boundary'],(int,float)) or not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):raise ValueError('Invalid physical metric boundary')
        p=read_arrays(prediction,schema,'Prediction')
        metrics=physical_metrics(p,ROOT.parent/'data/raw_data.npz')
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite physical metric')
        passed=all(metrics[k]<=thresholds[k]['boundary'] for k in metrics)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
    except Exception as error:report['errors']=[str(error)]
    return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':sys.exit(main())
