"""Independent full-image physical and standard-method evaluation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,110935,4),'soil_temperature':(1,110935),'canopy_temperature':(1,110929),'soil_heat_flux':(1,110935),'radiation':(1,110935,4)}
def validate(values,label):
    for key,shape in SHAPES.items():
        if key not in values:raise ValueError(f'{label}: missing {key}')
        a=values[key]
        if a.shape!=shape or a.dtype!=np.dtype('float64') or not np.isfinite(a).all():raise ValueError(f'{label}: {key} requires finite float64 {shape}')
    if np.any(values['soil_temperature']<=0) or np.any(values['canopy_temperature']<=0):raise ValueError(f'{label}: temperatures must be positive kelvin')
def evaluate(prediction,reference=None,configuration=None,observations=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        if reference is None:
            truth=HERE/'ground_truth.npz';asset=truth if truth.exists() else HERE/'baseline_reference.npz'
            with np.load(asset,allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        if configuration is None:configuration=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        if observations is None:
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
        validate(prediction,'prediction');validate(reference,'reference')
        ndvi=observations['NDVI'][0];tr=observations['radiometric_temperature'][0].astype('float64')
        if ndvi.shape!=(110935,) or tr.shape!=(110935,) or not np.isfinite(np.r_[ndvi,tr]).all():raise ValueError('evaluator: malformed public physical inputs')
        lai=-np.log(1-np.clip(ndvi-.05,0,1))/.5;mask=lai>0
        if mask.sum()!=110929:raise ValueError('evaluator: wrong observation support')
        # Independent nadir spherical-LAD/fc=1 view fraction, dimensionless.
        k=1/(1+1.774*(1+1.182)**(-.733));fraction=1-np.exp(-k*lai[mask].astype('float64'))
        ts=prediction['soil_temperature'][0];tc=prediction['canopy_temperature'][0]
        mixed=ts.copy();mixed[mask]=(fraction*tc**4+(1-fraction)*ts[mask]**4)**.25
        flux=prediction['reconstruction'][0];radiation=prediction['radiation'][0];g=prediction['soil_heat_flux'][0]
        closure_c=radiation[:,0]+radiation[:,2]-flux[:,0]-flux[:,1]
        closure_s=radiation[:,1]+radiation[:,3]-flux[:,2]-flux[:,3]-g
        m=dict(maximum_temperature_absolute_error=max(float(np.max(abs(prediction[k]-reference[k]))) for k in ['soil_temperature','canopy_temperature']),maximum_flux_absolute_error=max(float(np.max(abs(prediction[k]-reference[k]))) for k in ['reconstruction','soil_heat_flux','radiation']),maximum_radiometric_reprojection_K=float(np.max(abs(mixed-tr))),maximum_energy_closure_W_m2=max(float(np.max(abs(closure_c))),float(np.max(abs(closure_s)))))
        # Absent canopy is an acquisition branch, not a failed-fit zero fill.
        if np.any(flux[~mask,:2]!=0) or np.any(radiation[~mask][:,[0,2]]!=0):raise ValueError('Bare pixels require absent-canopy zero components')
        for key,value in m.items():
            bound=float(configuration[key+'_boundary'])
            if not np.isfinite([value,bound]).all() or bound<0:raise ValueError(f'evaluator: invalid metric/boundary {key}')
            report['thresholds'][key]=dict(comparison='<=',boundary=bound)
        report['metrics']=m;report['passed']=all(m[k]<=v['boundary'] for k,v in report['thresholds'].items());report['status']='pass' if report['passed'] else 'fail'
    except Exception as exc:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    try:
        with np.load(args.prediction,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        report=evaluate(values)
    except Exception as exc:report=dict(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'report write failed: {exc}',file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]
if __name__=='__main__':sys.exit(main())
