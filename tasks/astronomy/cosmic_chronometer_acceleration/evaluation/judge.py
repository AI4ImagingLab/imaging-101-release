"""Standalone deterministic full-contract evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,4,200),'hubble_rate':(1,4,200),'hubble_derivative':(1,4,200),'hubble_constant':(1,4),'point_covariance':(1,4,200,2,2)}
def validated(values,label):
    for key,shape in SHAPES.items():
        if key not in values:raise ValueError(f'{label}: missing {key}')
        a=values[key]
        if a.shape!=shape or a.dtype!=np.dtype('float64'):raise ValueError(f'{label}: {key} requires float64 {shape}')
        if not np.isfinite(a).all():raise ValueError(f'{label}: {key} contains nonfinite values')
    if np.any(values['hubble_rate']<=0) or np.any(values['hubble_constant']<=0):raise ValueError(f'{label}: expansion rates must be positive')
    c=values['point_covariance']
    if not np.allclose(c,np.swapaxes(c,-1,-2),rtol=1e-12,atol=1e-12):raise ValueError(f'{label}: covariance must be symmetric')
    if np.any(np.linalg.eigvalsh(c)<=0):raise ValueError(f'{label}: covariance must be positive definite')
def evaluate(prediction,reference=None,configuration=None,observations=None):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        if reference is None:
            truth=HERE/'ground_truth.npz'
            with np.load(truth if truth.exists() else HERE/'baseline_reference.npz',allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        if configuration is None:configuration=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        if observations is None:
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
        validated(prediction,'prediction');validated(reference,'reference')
        h=prediction['hubble_rate'];hp=prediction['hubble_derivative'];c=prediction['point_covariance'];r=reference;grid=np.linspace(0,1.965,200)
        scale=np.sqrt(np.diagonal(r['point_covariance'],axis1=-2,axis2=-1))
        z,y,e=[observations[k][0] for k in ['redshift','hubble_observation','hubble_error']]
        if z.shape!=(30,) or y.shape!=(30,) or e.shape!=(30,) or not np.isfinite(np.r_[z,y,e]).all() or np.any(e<=0):raise ValueError('evaluator: invalid observation calibration')
        m={'maximum_q_absolute_error':float(np.max(abs(prediction['reconstruction']-r['reconstruction']))),'maximum_H_relative_error':float(np.max(abs(h-r['hubble_rate'])/r['hubble_rate'])),'maximum_Hprime_sigma_normalized_error':float(np.max(abs(hp-r['hubble_derivative'])/scale[...,1])),'maximum_covariance_sigma_normalized_error':float(np.max(abs(c-r['point_covariance'])/(scale[..., :,None]*scale[...,None,:]))),'q_identity_error':float(np.max(abs(prediction['reconstruction']-((1+grid)*hp/h-1)))),'H0_identity_relative_error':float(np.max(abs(prediction['hubble_constant']-h[...,0])/h[...,0])),'observation_standardized_rms':max(float(np.sqrt(np.mean(((np.interp(z,grid,hh)-y)/e)**2))) for hh in h[0])}
        for key,value in m.items():
            bound=float(configuration[key+'_boundary'])
            if not np.isfinite(value) or not np.isfinite(bound) or bound<0:raise ValueError(f'evaluator: invalid metric or boundary {key}')
            report['thresholds'][key]={'comparison':'<=','boundary':bound}
        report['metrics']=m;report['passed']=all(m[k]<=v['boundary'] for k,v in report['thresholds'].items());report['status']='pass' if report['passed'] else 'fail'
    except Exception as exc:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try:
        with np.load(args.prediction,allow_pickle=False) as z:prediction={k:z[k] for k in z.files}
        report=evaluate(prediction)
    except Exception as exc:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]}
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'report write failed: {exc}',file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]
if __name__=='__main__':sys.exit(main())
