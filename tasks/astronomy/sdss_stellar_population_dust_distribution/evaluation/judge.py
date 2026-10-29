"""Standalone evaluator: never imports solver or upstream code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
SHAPES={'reconstruction':(1,198),'attenuation':(1,2308),'ebv':(1,1)}
def load_output(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing required key '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape '+key)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/64 '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            out[key]=a
    w=out['reconstruction']
    if np.min(w)<0 or abs(float(w.sum())-1)>1e-5:raise ValueError('Weights must be nonnegative and sum to one')
    if np.min(out['attenuation'])<=0:raise ValueError('Attenuation must be strictly positive')
    if not 0<=out['ebv'][0,0]<=1.5:raise ValueError('EBV outside declared range')
    return out
def predicted_flux(output,data):
    models=data['templates'][0]*output['attenuation'][0]
    median=np.median(models,axis=1)
    if np.any(median<=0):raise ValueError('Invalid model normalization')
    return output['reconstruction'][0]@(models/median[:,None]*np.median(data['flux'][0]))
def score(pred,ref,data):
    scale=float(np.ptp(ref['attenuation']))
    if scale<=0:raise ValueError('Undefined reference attenuation dynamic range')
    return {'weight_max_error':float(np.max(abs(pred['reconstruction']-ref['reconstruction']))),
      'attenuation_nrmse':float(np.sqrt(np.mean((pred['attenuation']-ref['attenuation'])**2))/scale),
      'ebv_absolute_error':float(abs(pred['ebv'][0,0]-ref['ebv'][0,0])),
      'mean_standardized_squared_residual':float(np.mean(((predicted_flux(pred,data)-data['flux'][0])/data['uncertainty'][0])**2))}
def physical_score(pred,data):
    a=pred['attenuation'][0].astype('float64');a=a/np.median(a)
    wave=data['wavelength'][0].astype('float64')/10000.
    k=np.where(wave>=.63,2.659*(-1.857+1.040/wave)+4.05,2.659*(-2.156+1.509/wave-.198/wave**2+.011/wave**3)+4.05)
    law=10**(-.4*float(pred['ebv'][0,0])*k);law=law/np.median(law)
    return {'mean_standardized_squared_residual':float(np.mean(((predicted_flux(pred,data)-data['flux'][0])/data['uncertainty'][0])**2)),
            'log_attenuation_tv':float(np.sum(abs(np.diff(np.log(a))))),
            'color_law_logrms':float(np.sqrt(np.mean((np.log(a)-np.log(law))**2)))}


def evaluate(prediction,evaluation_dir=None):
    folder=Path(evaluation_dir) if evaluation_dir else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        with np.load(folder.parent/'data/raw_data.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        pred=load_output(prediction);values=physical_score(pred,data)
        for key,value in values.items():
            bound=config[key+'_boundary']
            if not np.isfinite(bound) or bound<=0 or not np.isfinite(value):raise ValueError('Invalid metric/boundary '+key)
            report['thresholds'][key]={'comparison':'<=','boundary':bound}
        report['metrics']=values;report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in values.items());report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
