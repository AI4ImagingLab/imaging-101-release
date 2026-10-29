"""Independent standalone surface evaluator: no reference solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.fft import fft2,ifft2,fftfreq,fftshift,ifftshift
HERE=Path(__file__).resolve().parent
SHAPE=(1,641,641)
def compare(estimate,reference):
    ep=float(np.max(abs(estimate)));rp=float(np.max(abs(reference)))
    if rp==0 or np.ptp(reference)<=0:raise ValueError('degenerate evaluator reference')
    if ep==0:return {'ncc':0.,'nrmse':float(np.sqrt(np.mean(reference.astype(float)**2))/np.ptp(reference))}
    e=estimate.astype(float)/ep;r=reference.astype(float)/rp
    ncc=float(np.vdot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r)))
    common=max(ep,rp);error=e*(ep/common)-r*(rp/common)
    return {'ncc':float(np.clip(ncc,-1,1)),'nrmse':float(min(1e300,np.sqrt(np.mean(error**2))*min(common/float(np.ptp(reference)),1e300)))}

def observation_residual(height):
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:reference=z['reference'][0];image=z['measurements'][0]
    peaks=np.array([[276,369],[271,277]]);axes=[fftshift(fftfreq(n,1/(2*np.pi))) for n in (641,641)];pixel=np.array([[axes[0][p[0]],axes[1][p[1]]] for p in peaks]);spacing=.0044/(2*np.pi/np.mean(abs(pixel)));q=pixel/spacing
    radius=np.linalg.norm(peaks[0]-peaks[1])/2;row,col=np.indices((641,641));rf=fft2(reference);imf=fft2(image);observed=[]
    for p in peaks:
        mask=ifftshift((row-p[0])**2+(col-p[1])**2<radius**2);observed.append(-np.angle(ifft2(imf*mask)*np.conj(ifft2(rf*mask))))
    observed=np.array(observed,dtype=float)
    # A periodic height has zero mean slopes; per-carrier constant offsets are nuisance tilt.
    scale=max(1.,float(np.max(abs(height))))
    if scale>1e100:return float(np.pi)
    ky,kx=np.meshgrid(fftfreq(641,spacing/(2*np.pi)),fftfreq(641,spacing/(2*np.pi)),indexing='ij');ft=fft2(height)
    gx=ifft2(1j*kx*ft).real;gy=ifft2(1j*ky*ft).real
    heff=(1-1.0003/1.34)*sum(1.34*d/n for d,n in [(.057,1.0003),(.012,1.48899),(.043,1.34)])
    predicted=-heff*np.array([v[1]*gx+v[0]*gy for v in q]);circle=np.exp(1j*(predicted-observed));offset=np.angle(np.mean(circle,axis=(1,2)))
    error=np.angle(circle*np.exp(-1j*offset[:,None,None]));return float(np.sqrt(np.mean(error**2)))
def evaluate(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('missing reconstruction key')
        a=z['reconstruction']
    if a.shape!=SHAPE:raise ValueError(f'reconstruction shape must be {SHAPE}')
    if a.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError('reconstruction dtype must be float32 or float64')
    if not np.isfinite(a).all():raise ValueError('reconstruction contains nonfinite values')
    with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as z:r=z['reconstruction']
    if r.shape!=SHAPE or r.dtype not in [np.dtype('float32'),np.dtype('float64')] or not np.isfinite(r).all():raise ValueError('invalid evaluator reference')
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));metrics=compare(a,r);metrics['phase_residual']=observation_residual(a[0]);bounds={k:{'comparison':'>=' if k=='ncc' else '<=','boundary':config[k+'_boundary']} for k in ['ncc','nrmse','phase_residual']}
    passed=metrics['ncc']>=config['ncc_boundary'] and metrics['nrmse']<=config['nrmse_boundary'] and metrics['phase_residual']<=config['phase_residual_boundary']
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
