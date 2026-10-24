"""Independent standalone evaluator for both phases and projected induction."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent
KEYS=('phase_B','phase_E','Bx','By')
SHAPE=(1,512,512)
# CODATA2022 (SciPy1.17.1) constants, recorded independently of solver imports.
H=6.62607015e-34;HBAR=1.0545718176461565e-34;E=1.602176634e-19
M=9.1093837139e-31;C=299792458.;PIXEL=5.1066;DF=199065.;VOLTAGE=200000.
def load(path,keys=KEYS,shape=SHAPE,dtype=np.float64):
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key in keys:
            if key not in z:raise ValueError('Missing required key '+key)
            a=z[key]
            if a.shape!=shape or a.dtype!=dtype:raise ValueError(f'Invalid {key}: expected {dtype} {shape}, received {a.dtype} {a.shape}')
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            result[key]=a
    return result
def score(x,r):
    x=x.ravel();r=r.ravel();spread=np.ptp(r);rn=np.linalg.norm(r);xn=np.linalg.norm(x)
    if spread<=0 or rn<=0:raise ValueError('Degenerate reference')
    return float(np.dot(x,r)/(xn*rn)) if xn else 0.,float(np.sqrt(np.mean((x-r)**2))/spread)
def observation_metrics(prediction):
    raw=load(ASSETS.parent/'data/raw_data.npz',('unflipped','flipped'),(1,7,512,512),np.float32)
    a=raw['unflipped'][0];b=raw['flipped'][0]
    images=np.array([a[2],b[2],(a[3]+b[3])/2,a[4],b[4]],dtype=np.float32)
    flux=images.sum(axis=(1,2));images*= (flux.max()/flux)[:,None,None]
    images=images/images.max()+1e-9;i0=images[2]
    derivatives={'B':.5*((images[3]-images[0])-(images[4]-images[1])), 'E':.5*((images[3]-images[0])+(images[4]-images[1]))}
    fy,fx=np.meshgrid(np.fft.fftfreq(512,PIXEL),np.fft.fftfreq(512,PIXEL),indexing='ij')
    wavelength=H*1e9/np.sqrt(2*M*E)/np.sqrt(VOLTAGE+.5*E/M/C**2*VOLTAGE**2)
    result={}
    for label,difference in derivatives.items():
        difference-=difference.sum()/np.ones(difference.shape).sum()
        ft=np.fft.fft2(prediction['phase_'+label][0])
        gy=np.fft.ifft2(2j*np.pi*fy*ft).real;gx=np.fft.ifft2(2j*np.pi*fx*ft).real
        calculated=wavelength/(2*np.pi)*np.fft.ifft2(2j*np.pi*(fy*np.fft.fft2(i0*gy)+fx*np.fft.fft2(i0*gx))).real
        target=difference/(2*DF)
        result['observation_'+label]=float(np.linalg.norm(calculated-target)/np.linalg.norm(target))
    dy,dx=np.gradient(prediction['phase_B'][0],edge_order=2)
    bx=HBAR/E/PIXEL*1e18*dy;by=-HBAR/E/PIXEL*1e18*dx
    numerator=np.linalg.norm(prediction['Bx'][0]-bx)**2+np.linalg.norm(prediction['By'][0]-by)**2
    denominator=np.linalg.norm(bx)**2+np.linalg.norm(by)**2
    result['induction_consistency']=float(np.sqrt(numerator/denominator)) if denominator else 1.
    return result
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'))
        names=[k+'_'+m for k in KEYS for m in ['ncc','nrmse']]+['observation_B','observation_E','induction_consistency']
        limits={k:config[k+'_boundary'] for k in names}
        for k,v in limits.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Invalid boundary '+k)
            if (k.endswith('_ncc') and not -1<=v<=1) or (not k.endswith('_ncc') and v<0):raise ValueError('Boundary range '+k)
        gt=ASSETS/'ground_truth.npz';ref=load(gt if gt.is_file() else ASSETS/'baseline_reference.npz')
        x=load(Path(prediction))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics={}
            for k in KEYS:
                ncc,nrmse=score(x[k],ref[k]);metrics[k+'_ncc']=ncc;metrics[k+'_nrmse']=nrmse
            metrics.update(observation_metrics(x))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metrics')
        thresholds={k:dict(comparison='>=' if k.endswith('_ncc') else '<=',boundary=v) for k,v in limits.items()}
        passed=all(metrics[k]>=v if k.endswith('_ncc') else metrics[k]<=v for k,v in limits.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
