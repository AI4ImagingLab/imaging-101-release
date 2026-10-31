"""Standalone full-native finite-support stochastic-method evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ANGLES=(0,45,90);LENGTHS=(86491,83165,85604);SEEDS=(20261029,20261030,20261031)
def require(a,shape,label):
    if a.shape!=shape or a.dtype!=np.dtype('float64') or not np.isfinite(a).all():raise ValueError(f'{label} requires finite float64 {shape}')
def evaluate(prediction,reference=None,configuration=None,observations=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        if reference is None:
            truth=HERE/'ground_truth.npz';asset=truth if truth.exists() else HERE/'baseline_reference.npz'
            with np.load(asset,allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        if configuration is None:configuration=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        if observations is None:
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:observations={k:z[k] for k in z.files}
        for key in ['magnetic_field','temperature','polarization']:
            if key not in prediction:raise ValueError(f'missing {key}')
            require(prediction[key],(1,3),key)
        if np.any(prediction['magnetic_field']<=0) or np.any(prediction['temperature']<=0) or np.any((prediction['polarization']<0)|(prediction['polarization']>np.pi/2)):raise ValueError('Nonphysical parameter support')
        members=SEEDS
        if 'magnetic_field' in reference and 'parameters_0_20261029' not in reference:
            # Actual truth, if later supplied, has the public output contract.
            actual=reference;reference={};members=(0,)
            for key in ['magnetic_field','temperature','polarization']:require(actual[key],(1,3),'actual truth '+key)
            for i,(angle,length) in enumerate(zip(ANGLES,LENGTHS)):
                require(actual[f'reconstruction_{angle}'],(1,length),'actual truth response')
                reference[f'parameters_{angle}_0']=np.array([[actual['magnetic_field'][0,i]*1e4,actual['temperature'][0,i]-273.15,actual['polarization'][0,i]]])
                reference[f'response_{angle}_0']=actual[f'reconstruction_{angle}']
        elif 'magnetic_field' in reference:
            for key in ['magnetic_field','temperature','polarization']:require(reference[key],(1,3),'primary method '+key)
            for i,(angle,length) in enumerate(zip(ANGLES,LENGTHS)):
                require(reference[f'reconstruction_{angle}'],(1,length),'primary method response')
                primary=reference[f'parameters_{angle}_20261029'][0]
                if not np.array_equal(reference[f'reconstruction_{angle}'],reference[f'response_{angle}_20261029']):raise ValueError('evaluator: inconsistent primary response')
                expected=np.array([primary[0]*1e-4,primary[1]+273.15,primary[2]])
                actual=np.array([reference[k][0,i] for k in ['magnetic_field','temperature','polarization']])
                if not np.array_equal(actual,expected):raise ValueError('evaluator: inconsistent primary physical parameters')
        metrics={};thresholds={};passed=True
        for i,(angle,length) in enumerate(zip(ANGLES,LENGTHS)):
            key=f'reconstruction_{angle}'
            if key not in prediction:raise ValueError(f'missing {key}')
            require(prediction[key],(1,length),key)
            candidates=[]
            for seed in members:
                rp=reference[f'parameters_{angle}_{seed}'];rs=reference[f'response_{angle}_{seed}']
                require(rp,(1,3),'reference parameters');require(rs,(1,length),'reference response')
                if rp[0,0]<=0 or rp[0,1]+273.15<=0 or not 0<=rp[0,2]<=np.pi/2:raise ValueError('evaluator: invalid original physical support')
                values=[abs(prediction['magnetic_field'][0,i]-rp[0,0]*1e-4)/(rp[0,0]*1e-4),abs(prediction['temperature'][0,i]-(rp[0,1]+273.15)),abs(prediction['polarization'][0,i]-rp[0,2]),float(np.sqrt(np.mean((prediction[key]-rs)**2)))]
                names=['magnetic_field_relative_error','temperature_absolute_error','polarization_absolute_error','spectral_rmse']
                bounds=[float(configuration[n+'_boundary']) for n in names]
                if not np.isfinite(values+bounds).all() or any(b<=0 for b in bounds):raise ValueError('evaluator: invalid metric/boundary')
                candidates.append((max(v/b for v,b in zip(values,bounds)),seed,values,bounds))
            ratio,seed,values,bounds=min(candidates,key=lambda q:q[0])
            for name,value,bound in zip(names,values,bounds):
                metric=f'{name}_{angle}';metrics[metric]=float(value);thresholds[metric]=dict(comparison='<=',boundary=bound)
            label='nearest_complete_original_seed' if members==SEEDS else 'actual_truth_member'
            metrics[f'{label}_{angle}']=seed
            y=observations[f'transmission_{angle}'];require(y,(1,length),'observed transmission')
            rms=float(np.sqrt(np.mean((prediction[key]-y)**2)));bound=float(configuration['observation_rmse_boundaries'][str(angle)])
            if not np.isfinite([rms,bound]).all() or bound<=0:raise ValueError('evaluator: invalid observation score/boundary')
            metrics[f'observation_rmse_{angle}']=rms;thresholds[f'observation_rmse_{angle}']=dict(comparison='<=',boundary=bound)
            passed=passed and ratio<=1 and rms<=bound
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
    except Exception as exc:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    return report
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try:
        with np.load(args.prediction,allow_pickle=False) as z:p={k:z[k] for k in z.files}
        report=evaluate(p)
    except Exception as exc:report=dict(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)])
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'report write failed: {exc}',file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]
if __name__=='__main__':sys.exit(main())
