"""Independent standalone evaluator: complete complex field plus measured trace."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy import fftpack
from scipy.optimize import minimize_scalar

def align_field(estimate,reference,dw):
    """Continuous periodic translation, constant phase, optional conjugate reversal."""
    ne=np.linalg.norm(estimate);nr=np.linalg.norm(reference)
    if ne==0 or nr==0:return dict(complex_relative_l2=1.,coherent_overlap=0.,time_shift_seconds=0.,global_phase_radians=0.,conjugate_time_reversal=False)
    a=estimate/ne;b=reference/nr;n=a.size;k=np.arange(n)-n//2
    A=np.fft.fftshift(np.fft.ifft(a,norm='ortho'));B=np.fft.fftshift(np.fft.ifft(b,norm='ortho'));best=None
    for reverse in [False,True]:
        aa=A.conj() if reverse else A;z=B.conj()*aa;K=16*n
        j=int(np.argmax(abs(np.fft.ifft(z,n=K)*K)));q=2*np.pi*j/K
        opt=minimize_scalar(lambda p:-abs(np.sum(z*np.exp(1j*k*p))),bounds=(q-2*np.pi/K,q+2*np.pi/K),method='bounded',options={'xatol':1e-14})
        shifted=aa*np.exp(1j*k*opt.x);phase=np.angle(np.vdot(B,shifted));shifted*=np.exp(-1j*phase)
        field=np.fft.fft(np.fft.ifftshift(shifted),norm='ortho')*(ne/nr)
        row=dict(complex_relative_l2=float(np.linalg.norm(field-b)),coherent_overlap=float(min(1.,abs(np.vdot(B,shifted)))),time_shift_seconds=float(((opt.x+np.pi)%(2*np.pi)-np.pi)/dw),global_phase_radians=float(phase),conjugate_time_reversal=reverse)
        if best is None or row['complex_relative_l2']<best['complex_relative_l2']:best=row
    return best

def trace_error(field,observed,delays,dw):
    """Independent phase-corrected Fourier quadrature and fractional-delay SHG."""
    n=field.size;dt=2*np.pi/(n*dw);k=np.arange(n);t=(k-n//2)*dt;w=(k-n//2)*dw
    fr=dt*n/(2*np.pi)*np.exp(1j*k*t[0]*dw);fs=np.exp(1j*t*w[0]);br=np.exp(-1j*k*t[0]*dw);bs=dw*np.exp(-1j*t*w[0])
    F=lambda x:fr*fftpack.ifft(fs*x)
    B=lambda x:bs*fftpack.fft(br*x)
    if np.max(abs(field))==0:return float(np.sqrt(np.mean(observed**2))/observed.max())
    # Detector gain absorbs this purely numerical amplitude normalization.
    e=field/np.max(abs(field));a=F(e);delayed=B(np.exp(1j*delays[:,None]*w)*a);signal=delayed*e
    s=F(signal);T=s.real*s.real+s.imag*s.imag;den=np.sum(T*T)
    gain=np.sum(observed*T)/den if den>0 else 0.
    return float(np.sqrt(np.mean((observed-gain*T)**2))/observed.max())

def evaluate(prediction,assets=None):
    assets=Path(assets) if assets else Path(__file__).resolve().parent
    base=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        definition=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        thresholds=definition['thresholds'];base['thresholds']=thresholds
        if set(thresholds)!={'complex_relative_l2','coherent_overlap','trace_R','energy_relative_error'}:raise ValueError('Undefined metric configuration')
        for item in thresholds.values():
            if item['comparison'] not in ['<=','>='] or not np.isfinite(item['boundary']):raise ValueError('Invalid metric boundary/comparison')
        with np.load(assets/'baseline_reference.npz',allow_pickle=False) as f:reference=f['reconstruction'].copy()
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as f:observed=f['trace'][0];delays=f['delay'][0]
        if reference.shape!=(1,512) or reference.dtype!=np.complex128 or not np.isfinite(reference).all():raise ValueError('Invalid hidden baseline')
        dt=2*np.pi/(512*407474458795.6243)
        if abs(np.sum(abs(reference)**2)*dt-1)>1e-6:raise ValueError('Hidden baseline is not unit energy')
        if observed.shape!=(512,512) or delays.shape!=(512,) or not np.isfinite(observed).all() or not np.isfinite(delays).all() or observed.max()<=0:raise ValueError('Invalid public acquisition assets')
    except Exception as exc:base['errors']=['Evaluator assets: '+str(exc)];return base,2
    try:
        with np.load(Path(prediction),allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('Required key reconstruction missing')
            estimate=f['reconstruction'].copy()
        if estimate.shape!=(1,512):raise ValueError('reconstruction must have shape (1,512)')
        if estimate.dtype!=np.complex128:raise ValueError('reconstruction must have dtype complex128')
        if not np.isfinite(estimate).all():raise ValueError('reconstruction contains NaN or Infinity')
    except Exception as exc:base['errors']=['Prediction: '+str(exc)];return base,2
    try:
        dw=407474458795.6243;dt=2*np.pi/(512*dw);e=estimate[0];ref=reference[0]
        maximum=float(np.max(abs(e)))
        if maximum>1e140:
            metrics=dict(energy_relative_error=1e300,complex_relative_l2=1e300,coherent_overlap=0.,trace_R=trace_error(e,observed,delays,dw))
        else:
            energy=float(np.sum(e.real**2+e.imag**2)*dt);al=align_field(e,ref,dw)
            metrics={k:al[k] for k in ['complex_relative_l2','coherent_overlap']};metrics.update(energy_relative_error=abs(energy-1.),trace_R=trace_error(e,observed,delays,dw))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite evaluation metric')
        passed=all(metrics[k]<=v['boundary'] if v['comparison']=='<=' else metrics[k]>=v['boundary'] for k,v in thresholds.items())
        base.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics);return base,0 if passed else 1
    except Exception as exc:base.update(metrics={},errors=['Evaluation: '+str(exc)]);return base,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
