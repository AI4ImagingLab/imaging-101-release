"""Standalone deterministic evaluator. No solver imports or code execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SPECS={'reconstruction':((1,5,35,35),np.dtype('float64')),'arterial_transit_time':((1,5,35,35),np.dtype('float64'))}

def metric_pair(estimate,reference):
    """Uncentered cosine NCC and dynamic-range-normalized RMS error."""
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    scale=np.ptp(b);den=np.linalg.norm(a)*np.linalg.norm(b)
    if scale<=0 or np.linalg.norm(b)==0:raise ValueError('Reference has undefined metrics')
    return (float(np.dot(a,b)/den) if den>0 else 0.,float(np.sqrt(np.mean((a-b)**2))/scale))

def read_maps(path):
    if not Path(path).is_file():raise ValueError(f'Missing NPZ: {path}')
    with np.load(path,allow_pickle=False) as z:
        if not set(SPECS).issubset(z.files):raise ValueError('Missing required output keys')
        arrays={k:z[k] for k in SPECS}
    for key,(shape,dtype) in SPECS.items():
        a=arrays[key]
        if a.shape!=shape:raise ValueError(f'{key}: expected shape {shape}, got {a.shape}')
        if a.dtype!=dtype:raise ValueError(f'{key}: expected real float64, got {a.dtype}')
        if not np.isfinite(a).all():raise ValueError(f'{key}: nonfinite values')
    return arrays

def observation_residual(maps,raw_path):
    """Relative signal L2 residual on the fixed acquisition mask.

    Evaluator independently evaluates a relaxation-weighted finite bolus in
    broadcast form. Clinical CBF is converted to mL/g/ms; no gain alignment.
    """
    with np.load(raw_path,allow_pickle=False) as z:
        m0=z['m0'].astype(np.float64);obs=z['magnetization'].astype(np.float64);mask=z['mask']>0
    if m0.shape!=(1,5,35,35) or obs.shape!=(1,7,5,35,35) or mask.shape!=m0.shape:raise ValueError('Invalid public calibration shape')
    if not np.isfinite(m0).all() or not np.isfinite(obs).all():raise ValueError('Invalid public calibration values')
    for key,array in maps.items():
        if np.any(array<0):raise ValueError(f'{key}: physical field must be nonnegative')
        if np.any(array[~mask]!=0):raise ValueError(f'{key}: exterior mask values must be exactly zero')
    f=maps['reconstruction']/6000000.;arrival=maps['arterial_transit_time'];eff=1/(1/1650.+f/.98)
    signals=[]
    for duration,delay in zip([100.,100.,150.,150.,400.,800.,1800.],[170.,270.,370.,520.,670.,1070.,1870.]):
        end=duration+delay
        delivered=np.minimum(np.maximum(end-arrival,0),duration)
        washout=np.maximum(end-arrival-duration,0)
        with np.errstate(over='raise',divide='raise',invalid='raise'):
            signals.append(1.7*m0*f*eff*np.exp(-arrival/1650.)*(1-np.exp(-delivered/eff))*np.exp(-washout/eff))
    pred=np.stack(signals,axis=1);valid=np.broadcast_to(mask[:,None],obs.shape);den=np.linalg.norm(obs[valid])
    if den==0:raise ValueError('Zero observation norm')
    return float(np.linalg.norm((pred-obs)[valid])/den)

def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        asset_dir=Path(asset_dir)
        config=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'))
        thresholds=config['thresholds']
        required={f'{k}_{metric}' for k in SPECS for metric in ['ncc','nrmse']}|{'signal_relative_l2'}
        if set(thresholds)!=required:raise ValueError('Invalid metric configuration')
        for name,rule in thresholds.items():
            if rule['comparison'] not in ['>=','<='] or not np.isfinite(rule['boundary']):raise ValueError('Invalid metric boundary')
            if ('ncc' in name and rule['comparison']!='>=') or ('ncc' not in name and rule['comparison']!='<='):raise ValueError('Invalid metric comparison direction')
        report['thresholds']=thresholds
        reference_path=asset_dir/'ground_truth.npz'
        if not reference_path.exists():reference_path=asset_dir/'baseline_reference.npz'
        reference=read_maps(reference_path);estimate=read_maps(prediction)
        for key in SPECS:
            ncc,nrmse=metric_pair(estimate[key],reference[key]);report['metrics'][key+'_ncc']=ncc;report['metrics'][key+'_nrmse']=nrmse
        report['metrics']['signal_relative_l2']=observation_residual(estimate,asset_dir.parent/'data/raw_data.npz')
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite metric')
        passed=all(report['metrics'][k]>=r['boundary'] if r['comparison']=='>=' else report['metrics'][k]<=r['boundary'] for k,r in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed))
        return report,0 if passed else 1
    except Exception as error:
        report.update(status='error',passed=False,metrics={},errors=[str(error)])
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
