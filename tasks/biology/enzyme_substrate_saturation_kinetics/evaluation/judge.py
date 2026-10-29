"""Standalone evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,8),'maximum_rate':(1,8),'fitted_rate':(1,8,10)}
NAMES={'Km_max_relative_error','Vmax_max_relative_error','fitted_rate_max_absolute_error','output_forward_consistency_max_absolute_error'}
def load_arrays(path):
    if not Path(path).is_file():raise ValueError(f'Missing file: {path}')
    with np.load(path,allow_pickle=False) as a:
        d={}
        for k,shape in SHAPES.items():
            if k not in a:raise ValueError(f'Missing key: {k}')
            x=a[k]
            if x.shape!=shape:raise ValueError(f'{k}: expected shape {shape}, got {x.shape}')
            if x.dtype!=np.dtype('float64'):raise ValueError(f'{k}: expected real float64')
            if not np.isfinite(x).all():raise ValueError(f'{k}: non-finite values')
            d[k]=x
    return d
def metrics(p,r,s):
    if np.any(r['reconstruction']<=0) or np.any(r['maximum_rate']<=0):raise ValueError('Invalid reference physical scale')
    if np.any(p['reconstruction']<=0) or np.any(p['maximum_rate']<=0):raise ValueError('Km and Vmax must be positive')
    calculated=p['maximum_rate'][...,None]*s/(p['reconstruction'][...,None]+s)
    return {'Km_max_relative_error':float(np.max(abs(p['reconstruction']/r['reconstruction']-1))),
      'Vmax_max_relative_error':float(np.max(abs(p['maximum_rate']/r['maximum_rate']-1))),
      'fitted_rate_max_absolute_error':float(np.max(abs(p['fitted_rate']-r['fitted_rate']))),
      'output_forward_consistency_max_absolute_error':float(np.max(abs(p['fitted_rate']-calculated)))}
def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8')); thresholds=cfg['thresholds']
        if set(thresholds)!=NAMES or any(v['comparison']!='<=' or not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):raise ValueError('Invalid metric configuration')
        report['thresholds']=thresholds
        ref=HERE/'ground_truth.npz'
        if not ref.exists():ref=HERE/'baseline_reference.npz'
        p=load_arrays(path);r=load_arrays(ref)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:s=a['substrate']
        if s.shape!=(1,10) or not np.isfinite(s).all() or np.any(s<=0):raise ValueError('Invalid public concentration calibration')
        if np.any(r['reconstruction']<=0) or np.any(r['maximum_rate']<=0):raise ValueError('Invalid reference physical scale')
        expected=r['maximum_rate'][...,None]*s/(r['reconstruction'][...,None]+s)
        if np.max(abs(expected-r['fitted_rate']))>1e-8:raise ValueError('Reference forward inconsistency')
        values=metrics(p,r,s)
        if not all(np.isfinite(x) for x in values.values()):raise ValueError('Non-finite metrics')
        passed=all(values[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return report,0 if passed else 1
    except Exception as e:
        report['errors']=[str(e)];report['metrics']={};return report,2
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);a=parser.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as e:print(f'Report write failed: {e}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
