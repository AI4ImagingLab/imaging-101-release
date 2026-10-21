"""Standalone evaluator: no solver or submission-code imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,512,512)
def compare(estimate,reference):
    # Normalize separately for robust cosine and constant-phase alignment.
    ep=float(np.max(np.abs(estimate.real)));ep=max(ep,float(np.max(np.abs(estimate.imag))))
    rp=max(float(np.max(np.abs(reference.real))),float(np.max(np.abs(reference.imag))))
    if rp==0:raise ValueError('reference norm is zero')
    if ep==0:return {'ncc':0.,'nrmse':float(np.sqrt(np.mean(abs(reference)**2))/np.ptp(abs(reference)))}
    e=estimate/ep;r=reference/rp
    dot=np.vdot(r,e);phase=np.exp(-1j*np.angle(dot))
    ncc=float(abs(dot)/(np.linalg.norm(e)*np.linalg.norm(r)))
    common=max(ep,rp);difference=(e*(ep/common)*phase-r*(rp/common))
    dr=float(np.ptp(abs(reference)))
    if dr<=0:raise ValueError('reference amplitude dynamic range is zero')
    nr=float(np.sqrt(np.mean(abs(difference)**2)))
    # Saturation for extreme but finite valid predictions keeps JSON finite.
    nrmse=float(min(1e300,nr*min(common/dr,1e300)))
    return {'ncc':min(1.,ncc),'nrmse':nrmse}
def evaluate(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('missing reconstruction key')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError(f'reconstruction shape must be {SHAPE}')
    if a.dtype not in [np.dtype('complex64'),np.dtype('complex128')]:raise ValueError('reconstruction dtype must be complex64 or complex128')
    if not np.isfinite(a).all():raise ValueError('reconstruction contains nonfinite values')
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:r=z['reconstruction']
    if r.shape!=SHAPE or not np.isfinite(r).all():raise ValueError('invalid evaluator reference')
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
    bounds={'ncc':{'comparison':'>=','boundary':config['ncc_boundary']},'nrmse':{'comparison':'<=','boundary':config['nrmse_boundary']}}
    metrics=compare(a,r)
    # Independent forward observation check using the declared acquisition constants.
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:d=z['measurements'][0].astype('float32')/65535
    low=d.min(axis=(-2,-1),keepdims=True);amp=np.sqrt((d-low)/(1-low))
    axes=[np.fft.fftshift(np.linspace(-n/2/(1.12*n)*2*np.pi,(n/2-1)/(1.12*n)*2*np.pi,n)) for n in (512,512)]
    g=np.array(np.meshgrid(*axes,indexing='ij'),dtype='float32');kt2=np.sum(g*g,axis=0);delta=(2*np.pi/.532)**2-kt2
    scale=max(1.,float(np.max(abs(a.real))),float(np.max(abs(a.imag))))
    normalized=a[0]/scale
    pred=np.stack([abs(np.fft.ifft2(np.exp(1j*(np.sqrt(delta)*np.float64(z)))*np.fft.fft2(normalized))) for z in range(300,651,50)])
    residual=float(min(1e300,np.linalg.norm(pred-amp/scale)/np.linalg.norm(amp.astype(np.float64))*min(scale,1e300)))
    metrics['amplitude_residual']=residual
    bounds['amplitude_residual']={'comparison':'<=','boundary':config['amplitude_residual_boundary']}
    passed=metrics['ncc']>=bounds['ncc']['boundary'] and metrics['nrmse']<=bounds['nrmse']['boundary'] and residual<=bounds['amplitude_residual']['boundary']
    return {'status':'pass' if passed else 'fail','passed':bool(passed),'metrics':metrics,'thresholds':bounds,'errors':[]}
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    try:result=evaluate(Path(args.prediction));code=0 if result['passed'] else 1
    except Exception as exc:result={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print(f'report write failed: {exc}',file=sys.stderr);return 2
    print(json.dumps(result,allow_nan=False));return code
if __name__=='__main__':raise SystemExit(main())
