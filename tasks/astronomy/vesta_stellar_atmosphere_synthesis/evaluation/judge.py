"""Independent stellar-parameter/spectrum evaluator, with no solver imports."""
from pathlib import Path
import argparse,json,sys,numpy as np
HERE=Path(__file__).resolve().parent

def prediction(path):
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for key,shape in [('reconstruction',(1,4)),('spectrum',(1,52570))]:
            if key not in z:raise ValueError('Missing '+key)
            a=z[key]
            if a.shape!=shape or a.dtype!=np.dtype('float64'):raise ValueError(key+' requires real float64 '+str(shape))
            if not np.isfinite(a).all():raise ValueError(key+' contains nonfinite values')
            arrays[key]=a
    return arrays

def evaluate(path,assets=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(assets);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        names=['parameter_standardized_max','spectrum_nrmse','masked_weighted_rmse']
        boundaries={name:config[name+'_boundary'] for name in names}
        if not all(np.isfinite(v) and v>0 for v in boundaries.values()):raise ValueError('Invalid metric boundary')
        reference=assets/'ground_truth.npz'
        if not reference.exists():reference=assets/'baseline_reference.npz'
        ref=prediction(reference);estimate=prediction(path)
        # These standard-method weighting arrays remain identified measurements,
        # never acquisition calibration or a substitute for physical truth.
        with np.load(assets/'baseline_reference.npz',allow_pickle=False) as z:scale=z['parameter_scale'];noise=z['noise_reference']
        if scale.shape!=(1,4) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid parameter scales')
        raw=assets.parent/'data/raw_data.npz'
        with np.load(raw,allow_pickle=False) as z:wave=z['wavelength'][0];flux=z['flux'][0];masks=z['masks'][0]
        indices=np.concatenate([np.flatnonzero(np.abs(wave-center)<=width) for center,width in masks])
        if noise.shape!=(1,len(indices)) or not np.isfinite(noise).all() or np.any(noise<=0):raise ValueError('Invalid reference noise weights')
        dynamic=float(np.ptp(ref['spectrum']))
        if not np.isfinite(dynamic) or dynamic<=0:raise ValueError('Undefined constant-spectrum NRMSE')
        metrics=dict(parameter_standardized_max=float(np.max(abs(estimate['reconstruction']-ref['reconstruction'])/scale)),spectrum_nrmse=float(np.sqrt(np.mean((estimate['spectrum']-ref['spectrum'])**2))/dynamic),masked_weighted_rmse=float(np.sqrt(np.mean(((estimate['spectrum'][0,indices]-flux[indices])/noise[0])**2))))
        passed=all(metrics[n]<=boundaries[n] for n in names)
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds={name:dict(comparison='<=',boundary=boundaries[name]) for name in names})
    except Exception as exc:report['errors']=[str(exc)]
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(str(exc),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[report['status']]

if __name__=='__main__':sys.exit(main())
