"""Standalone deterministic evaluator; never imports reference/solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np

SCALARS=('V_P','V_F','I_es','I_is','n_e','n_i','T_e','n_i_OML')
SHAPES={**{k:(1,1) for k in SCALARS},'reconstruction':(1,64),'energy':(1,64)}

def load_arrays(path):
    if not Path(path).is_file():raise ValueError(f'Missing NPZ file: {path}')
    arrays={}
    with np.load(path,allow_pickle=False) as data:
        for key,shape in SHAPES.items():
            if key not in data:raise ValueError(f'Missing key: {key}')
            a=data[key]
            if a.shape!=shape:raise ValueError(f'{key}: expected shape {shape}, got {a.shape}')
            if a.dtype!=np.dtype('float64'):raise ValueError(f'{key}: expected float64')
            if not np.isfinite(a).all():raise ValueError(f'{key}: nonfinite values')
            arrays[key]=a.copy()
    return arrays

def evaluate(prediction,asset_dir=None):
    assets=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        settings=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        ref=load_arrays(assets/'baseline_reference.npz')
        p=load_arrays(prediction)
        thresholds={k:{'comparison':'<=','boundary':settings[k+'_boundary']} for k in ['eedf_relative_l2','energy_max_abs','parameter_max_relative','normalization_error']}
        if any(not np.isfinite(x['boundary']) or x['boundary']<0 for x in thresholds.values()):raise ValueError('Invalid metric boundary')
        norm=np.linalg.norm(ref['reconstruction'])
        if norm<=0 or any(np.any(ref[k]==0) for k in SCALARS):raise ValueError('Degenerate reference metric scale')
        measured=dict(eedf_relative_l2=float(np.linalg.norm(p['reconstruction']-ref['reconstruction'])/norm),energy_max_abs=float(np.max(abs(p['energy']-ref['energy']))),parameter_max_relative=max(float(np.max(abs(p[k]-ref[k])/abs(ref[k]))) for k in SCALARS),normalization_error=float(abs(abs(np.trapezoid(p['reconstruction'][0],x=p['energy'][0]))-1)))
        passed=all(measured[k]<=thresholds[k]['boundary'] for k in measured)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
