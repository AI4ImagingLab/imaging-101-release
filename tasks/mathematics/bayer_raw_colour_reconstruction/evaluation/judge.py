"""Standalone deterministic evaluator; never imports submitted or reference solver."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent

def image_metrics(image,reference):
    """Uncentered cosine NCC and reference-dynamic-range NRMSE."""
    x=np.asarray(image).ravel();r=np.asarray(reference).ravel()
    nr=float(np.linalg.norm(r));nx=float(np.linalg.norm(x));dr=float(np.max(r)-np.min(r))
    if nr<=0 or dr<=0 or not np.isfinite([nr,nx,dr]).all():raise ValueError('Undefined reference or overflowing metric')
    ncc=float(np.dot(x,r)/(nx*nr)) if nx>0 else 0.0
    nrmse=float(np.sqrt(np.mean((x-r)**2))/dr)
    if not np.isfinite([ncc,nrmse]).all():raise ValueError('Nonfinite metric')
    return {'ncc':ncc,'nrmse':nrmse}

def read_array(path,shape):
    with np.load(path,allow_pickle=False) as z:a=z['reconstruction']
    if a.shape!=tuple(shape) or a.dtype!=np.float64:raise ValueError('Invalid reconstruction shape/dtype')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a

def evaluate(prediction,asset_dir=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir);c=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        if set(c['outputs'])!={'reconstruction'}:raise ValueError('Invalid output specification')
        s=c['outputs']['reconstruction']
        if s['dtype']!='float64' or len(s['shape'])!=4 or s['shape'][0]!=1 or s['shape'][-1]!=3:raise ValueError('Invalid evaluator shape/dtype configuration')
        lo=float(c['ncc_boundary']);hi=float(c['nrmse_boundary']);obs=float(c['sample_consistency_max_abs_boundary'])
        if not np.isfinite([lo,hi,obs]).all() or not(-1<=lo<=1) or hi<=0 or obs<=0:raise ValueError('Invalid metric boundaries')
        path=assets/'ground_truth.npz'
        if not path.exists():path=assets/'baseline_reference.npz'
        reference=read_array(path,s['shape']);prediction=read_array(prediction,s['shape'])
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z:raw=z['sensor_mosaic'];black=z['black_levels']
        if raw.shape!=tuple(s['shape'][:3]) or raw.dtype!=np.uint16 or black.shape!=(1,4) or black.dtype!=np.float64:raise ValueError('Invalid evaluator calibration shapes/dtypes')
        if not np.isfinite(black).all() or np.any(black<0) or np.any(black>=1023) or black[0,1]!=black[0,3]:raise ValueError('Invalid black calibration')
        if c['acquisition']!={'pattern':'RGGB','white_level':1023}:raise ValueError('Acquisition configuration mismatch')
        values=image_metrics(prediction,reference)
        measured=np.empty(raw.shape,dtype=np.float64)
        for dr,dc,ch,bi in [(0,0,0,0),(0,1,1,1),(1,0,1,3),(1,1,2,2)]:
            target=(raw[:,dr::2,dc::2].astype(np.float64)-black[0,bi])/(1023-black[0,bi])
            measured[:,dr::2,dc::2]=prediction[:,dr::2,dc::2,ch]-target
        values['sample_consistency_max_abs']=float(np.max(np.abs(measured)))
        if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite metric')
        thresholds={'ncc':{'comparison':'>=','boundary':lo},'nrmse':{'comparison':'<=','boundary':hi},'sample_consistency_max_abs':{'comparison':'<=','boundary':obs}}
        passed=values['ncc']>=lo and values['nrmse']<=hi and values['sample_consistency_max_abs']<=obs
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
