"""Standalone calibrated-MPI evaluator; no submitted or reference solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HOME=Path(__file__).resolve().parent

def validate(z):
    if 'reconstruction' not in z:raise ValueError('Missing reconstruction key')
    x=z['reconstruction']
    if x.shape!=(1,19,19,19) or x.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 reconstruction shape(1,19,19,19)')
    if not np.isfinite(x).all():raise ValueError('Reconstruction contains NaN or infinity')
    if np.any(x<0):raise ValueError('Tracer amounts must be nonnegative')

def observation_residual(coefficients, ridge=False):
    """Chunked independent calibrated forward prediction from public raw arrays."""
    with np.load(HOME.parent/'data/raw_data.npz',allow_pickle=False) as z:
        cal=z['calibration'][0]
        if cal.shape!=(7116,6859) or cal.dtype!=np.complex64:raise ValueError('Invalid public calibration')
        bg=z['calibration_background'][0]
        if bg.shape!=(7116,362) or bg.dtype!=np.complex64:raise ValueError('Invalid calibration backgrounds')
        offset=bg.mean(axis=1);del bg
        fg=z['foreground'][0]
        if fg.shape!=(1000,7116) or fg.dtype!=np.complex128:raise ValueError('Invalid foreground cycles')
        u=fg.mean(axis=0);del fg
        bg=z['background'][0]
        if bg.shape!=(1000,7116) or bg.dtype!=np.complex128:raise ValueError('Invalid background cycles')
        u-=bg.mean(axis=0);del bg
    energy=0.
    x=coefficients.ravel();fitted=np.empty(7116,dtype=np.complex128)
    for start in range(0,7116,64):
        stop=min(start+64,7116)
        # Keep the nativecomplex64 calibration subtraction convention.
        block=cal[start:stop]-offset[start:stop,None]
        fitted[start:stop]=block@x
        energy+=float(np.sum(np.abs(block.astype(np.complex128))**2))
    denom=np.linalg.norm(u)
    if denom==0:raise ValueError('Degenerate public observation')
    residual=float(np.linalg.norm(fitted-u)/denom)
    return float((np.linalg.norm(fitted-u)**2+0.001*np.sqrt(energy)*np.dot(x,x))/denom**2) if ridge else residual

def reference_metrics(x,ref):
    a=x.ravel();b=ref.ravel();norm=np.linalg.norm(a)*np.linalg.norm(b);span=b.max()-b.min()
    if np.linalg.norm(b)==0 or span<=0:raise ValueError('Degenerate evaluator reference')
    return {'ncc_vs_ref':float(np.dot(a,b)/norm) if norm>0 else 0.,'nrmse_vs_ref':float(np.sqrt(np.mean((a-b)**2))/span)}

def evaluate(prediction,report):
    result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        with np.load(prediction,allow_pickle=False) as z:pred={k:z[k] for k in z.files}
        validate(pred)
        asset=HOME/('ground_truth.npz' if (HOME/'ground_truth.npz').exists() else 'baseline_reference.npz')
        with np.load(asset,allow_pickle=False) as z:ref={k:z[k] for k in z.files}
        validate(ref)
        if not np.any(ref['reconstruction']):raise ValueError('Degenerate standard coefficients')
        config=json.loads((HOME/'metrics.json').read_text(encoding='utf-8'))
        boundary=config['objective_ratio_boundary']
        if isinstance(boundary,bool) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid objective ratio boundary')
        objective=observation_residual(pred['reconstruction'],ridge=True)
        attainable=observation_residual(ref['reconstruction'],ridge=True)
        if attainable<=0 or not np.isfinite(attainable):raise ValueError('Degenerate standard objective')
        metrics={'ridge_objective_ratio':objective/attainable}
        thresholds={'ridge_objective_ratio':{'comparison':'<=','boundary':boundary}}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined metric')
        passed=metrics['ridge_objective_ratio']<=boundary
        result.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:result['errors']=[f'{type(exc).__name__}: {exc}']
    try:
        target=Path(report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prediction',required=True);ap.add_argument('--report',required=True);a=ap.parse_args();sys.exit(evaluate(a.prediction,a.report))
