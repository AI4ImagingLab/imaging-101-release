"""Independent AMARES-quantity/observation evaluator; never imports solver code."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ASSETS=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,16),'frequency_hz':(1,16),'linewidth_hz':(1,16),'phase_rad':(1,1)}
METRICS={'amplitude_ncc':'>=','amplitude_nrmse':'<=','frequency_max_abs_hz':'<=',
         'linewidth_max_abs_hz':'<=','phase_periodic_abs_rad':'<=','observation_relative_l2':'<='}


def load_prediction(path):
    if not Path(path).is_file():raise ValueError(f'Missing NPZ file: {path}')
    with np.load(path,allow_pickle=False) as f:
        missing=set(SHAPES)-set(f.files)
        if missing:raise ValueError(f'Missing required keys: {sorted(missing)}')
        arrays={k:f[k] for k in SHAPES}
    for key,value in arrays.items():
        if value.shape!=SHAPES[key]:raise ValueError(f'{key}: expected shape {SHAPES[key]}, got {value.shape}')
        if value.dtype!=np.dtype('float64'):raise ValueError(f'{key}: expected real float64, got {value.dtype}')
        if not np.isfinite(value).all():raise ValueError(f'{key}: nonfinite values')
    if (arrays['reconstruction']<0).any():raise ValueError('Resonance amplitudes must be nonnegative')
    if (arrays['linewidth_hz']<=0).any():raise ValueError('Linewidths must be positive')
    if (abs(arrays['phase_rad'])>np.pi).any():raise ValueError('Common receiver phase must be within [-pi,pi]')
    return arrays


def evaluate(prediction,assets=ASSETS):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        for name,direction in METRICS.items():
            threshold=config[name+'_boundary']
            if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<=0:
                raise ValueError(f'Invalid evaluator threshold {name}')
            if name=='amplitude_ncc' and threshold>1:raise ValueError('NCC boundary exceeds one')
            report['thresholds'][name]=dict(comparison=direction,boundary=threshold)
        truth=assets/'ground_truth.npz'
        ref=load_prediction(truth if truth.is_file() else assets/'baseline_reference.npz')
        r=ref['reconstruction'].ravel()
        if np.ptp(r)<=0 or np.linalg.norm(r)==0:raise ValueError('Degenerate amplitude reference')
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as f:
            fid=f['measurements']
        if fid.shape!=(1,1024) or fid.dtype!=np.dtype('complex128') or not np.isfinite(fid).all():
            raise ValueError('Invalid evaluator observation/calibration asset')
        if np.linalg.norm(fid)==0:raise ValueError('Zero evaluator observation norm')
        p=load_prediction(prediction);x=p['reconstruction'].ravel()
        norm=np.linalg.norm(x)*np.linalg.norm(r)
        time=np.arange(0,1024/10000.,1/10000.)+300e-6
        amplitude=x[:,None];frequency=p['frequency_hz'].ravel()[:,None]
        width=p['linewidth_hz'].ravel()[:,None];phase=float(p['phase_rad'][0,0])
        modeled=np.sum(amplitude*np.exp(1j*phase)*np.exp(-np.pi*width*time)
                       *np.exp(2j*np.pi*frequency*time),axis=0)
        metrics=dict(amplitude_ncc=float(x@r/norm) if norm else 0.,
            amplitude_nrmse=float(np.sqrt(np.mean((x-r)**2))/np.ptp(r)),
            frequency_max_abs_hz=float(np.max(abs(p['frequency_hz']-ref['frequency_hz']))),
            linewidth_max_abs_hz=float(np.max(abs(p['linewidth_hz']-ref['linewidth_hz']))),
            phase_periodic_abs_rad=float(abs(np.angle(np.exp(1j*(phase-ref['phase_rad'][0,0]))))),
            observation_relative_l2=float(np.linalg.norm(modeled-fid[0])/np.linalg.norm(fid[0])))
        if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite evaluation metric')
        passed=all(v>=report['thresholds'][k]['boundary'] if METRICS[k]=='>='
                   else v<=report['thresholds'][k]['boundary'] for k,v in metrics.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics)
        return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(Path(args.prediction))
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr);return 2
    return code


if __name__=='__main__':raise SystemExit(main())
