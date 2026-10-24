"""Standalone evaluator; independent NumPy math, no src imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ASSETS=Path(__file__).resolve().parent;SHAPE=(1,50,50);BIN_MM=.149896229
def load_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        a=z['reconstruction']
    if a.shape!=SHAPE or a.dtype!=np.float64:raise ValueError('Require float64 reconstruction(1,50,50)')
    if not np.isfinite(a).all():raise ValueError('Nonfinite reconstruction')
    return a
def correlation(y,h):
    y=np.asarray(y,dtype=np.float64);y-=y.mean(axis=-1,keepdims=True);y/=np.linalg.norm(y,axis=-1,keepdims=True)+1e-8
    h=h-h.mean();h=h/(np.linalg.norm(h)+1e-8)
    return np.fft.ifft(np.conj(np.fft.fft(h))*np.fft.fft(y,axis=-1),axis=-1).real
def score(x,r):
    a=np.abs(x-r)/BIN_MM
    return {'max_abs_bin_error':float(a.max()),'mean_abs_bin_error':float(a.mean())}
def observation_regret(x):
    with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as z:counts=z['photon_counts'];h=z['instrument_response']
    if counts.shape!=(1,50,50,10000) or counts.dtype!=np.uint32 or h.shape!=(1,10000) or h.dtype!=np.float64 or not np.isfinite(h).all() or (h<0).any() or not np.isfinite(h.sum()) or h.sum()<=0:raise ValueError('Invalid public calibration/observation')
    h=np.roll(h[0],-int(np.argmax(h[0])));h=h/h.sum();reference=int(np.argmax(correlation(counts[0,0,0].copy(),h)))
    bins=x[0]/BIN_MM+reference
    if abs(float(x[0,0,0]))>1e-8 or np.max(abs(bins-np.rint(bins)))>1e-5:raise ValueError('Output must use integer timing bins and zero corner gauge')
    if bins.min()<0 or bins.max()>=10000:raise ValueError('Delay outside acquired timestamp window')
    bins=np.rint(bins).astype(int);total=0.
    for row in range(50):
        c=correlation(counts[0,row].copy(),h);chosen=c[np.arange(50),bins[row]];total+=np.maximum(0,c.max(axis=-1)-chosen).sum()
    return float(total/2500)
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'));names=['max_abs_bin_error','mean_abs_bin_error','mean_correlation_regret'];limits={k:cfg[k+'_boundary'] for k in names}
        for k,v in limits.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 or (k=='mean_correlation_regret' and v>2):raise ValueError('Invalid boundary '+k)
        truth=ASSETS/'ground_truth.npz';r=load_prediction(truth if truth.is_file() else ASSETS/'baseline_reference.npz');x=load_prediction(prediction)
        with np.errstate(over='raise',invalid='raise',divide='raise'):metrics=score(x,r);metrics['mean_correlation_regret']=observation_regret(x)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
        passed=all(metrics[k]<=limits[k] for k in names);report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds={k:{'comparison':'<=','boundary':v} for k,v in limits.items()});return report,0 if passed else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv);r,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
