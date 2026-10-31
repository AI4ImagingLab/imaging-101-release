"""Standalone full-cube evaluator; no submitted or reference-solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,64,64,2048)

def load_cube(path):
    if not Path(path).is_file():raise ValueError('Missing prediction/reference NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
        x=f['reconstruction']
        if x.shape!=SHAPE:raise ValueError('Expected shape'+str(SHAPE))
        if x.dtype!=np.dtype('float64'):raise ValueError('Expected real float64')
        if not np.isfinite(x).all():raise ValueError('Nonfinite cube')
        return x

def reference_metrics(x,ref):
    span=float(np.ptp(ref));refnorm=float(np.linalg.norm(ref))
    if span<=0 or refnorm==0:raise ValueError('Degenerate evaluator reference')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        norm=float(np.linalg.norm(x))
        ncc=0. if norm==0 else float(np.vdot(x,ref).real/(norm*refnorm))
        nrmse=float(np.sqrt(np.mean((x-ref)**2))/span)
    if not np.isfinite([ncc,nrmse]).all():raise ValueError('Metric overflow or undefined value')
    return dict(ncc=float(np.clip(ncc,-1,1)),nrmse=nrmse)

def observation_residual(cube,data_path=HERE.parent/'data/raw_data.npz'):
    """Independent dense64x64 Walsh sensing, chunked across spectral channels."""
    with np.load(data_path,allow_pickle=False) as f:
        raw=f['measurements'];ids=f['pattern_id']
    if raw.shape!=(1,8192,2048) or raw.dtype!=np.float64 or ids.shape!=(1,8192) or ids.dtype!=np.int64:raise ValueError('Malformed public observation/calibration')
    if not np.isfinite(raw).all() or not np.array_equal(np.sort(ids[0]),np.arange(8192)):raise ValueError('Invalid public observation/calibration')
    # Independently construct Sylvester blocks, then count sign changes to sort
    # rows by sequency. This does not copy the solver Gray-code permutation.
    w=np.array([[1.]],dtype=float)
    while len(w)<64:w=np.block([[w,w],[w,-w]])
    crossings=np.count_nonzero(np.diff(w,axis=1),axis=1)
    w=w[np.argsort(crossings)]
    total_error=0.;total_signal=0.
    with np.errstate(over='ignore',invalid='ignore'):
        for start in range(0,2048,128):
            x=cube[0,...,start:start+128]
            temp=np.einsum('ai,ijc->ajc',w,x,optimize=True)
            predicted=np.einsum('ajc,bj->abc',temp,w,optimize=True).reshape(4096,-1)[ids[0,::2]//2]
            observed=raw[0,::2,start:start+128]-raw[0,1::2,start:start+128]
            total_error+=float(np.sum((predicted-observed)**2));total_signal+=float(np.sum(observed**2))
    if total_signal<=0 or not np.isfinite([total_error,total_signal]).all():raise ValueError('Undefined forward consistency metric')
    return float(np.sqrt(total_error/total_signal))

def evaluate(prediction,asset_dir=HERE):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(asset_dir);cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        ncc=cfg['ncc_boundary'];nrmse=cfg['nrmse_boundary'];res=cfg['relative_measurement_residual_boundary']
        if not np.isfinite([ncc,nrmse,res]).all() or not -1<=ncc<=1 or nrmse<=0 or res<=0:raise ValueError('Invalid evaluator boundaries')
        refpath=assets/'ground_truth.npz'
        if not refpath.exists():refpath=assets/'baseline_reference.npz'
        reference=load_cube(refpath);x=load_cube(prediction)
        values=reference_metrics(x,reference)
        values['relative_measurement_residual']=observation_residual(x)
        thresholds=dict(ncc=dict(comparison='>=',boundary=ncc),nrmse=dict(comparison='<=',boundary=nrmse),relative_measurement_residual=dict(comparison='<=',boundary=res))
        passed=values['ncc']>=ncc and values['nrmse']<=nrmse and values['relative_measurement_residual']<=res
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    result,code=evaluate(a.prediction)
    try:
        out=Path(a.report);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
