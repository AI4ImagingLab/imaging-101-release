# EXAFS equation and transform adapted from xraylarch, MIT; see task LICENSE.
# Copyright 2010-2026 Matthew Newville and collaborators.
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import constants
REFF=2.5478
DEGENERACY=12.
KSTEP=.05
NFFT=2048
RSTEP=np.pi/(KSTEP*NFFT)
R=np.linspace(1.8,3.4,801)
INITIAL=np.array([1.,0.,0.,0.])
PARAMETER_SCALES=np.array([1.,1.,.01,.1])
ETOK=1./(1.e20*constants.hbar**2/(2*constants.m_e*constants.e))
def model_grid(data):
    return .05*np.arange(int(1.01+data['k'].max()/.05))
def scattering_splines(data):
    """Exact cubic, zero smoothing FEFF interpolation, including extrapolation."""
    return tuple(UnivariateSpline(data['scattering_k'],data[key],s=0) for key in ('scattering_phase','scattering_amplitude','real_momentum','mean_free_path'))
def forward(p,k,splines):
    """First-shell chi(k), with energy shift and full complex second-cumulant correction."""
    amp,e0,sigma2,dr=p
    en=k*k-e0*ETOK
    if np.min(np.abs(en))<1.e-6:en=np.where(np.abs(en)<1.5e-6,1.e-6,en)
    q=np.sign(en)*np.sqrt(np.abs(en))
    phase,f,rep,lam=(s(q) for s in splines)
    pp=(rep+1j/lam)**2
    momentum=np.sqrt(pp)
    cchi=np.exp(-2*REFF*momentum.imag-2*pp*sigma2+1j*(2*q*REFF+phase+2*momentum*(dr-2*sigma2/REFF)))
    cchi=DEGENERACY*amp*f*cchi/(q*(REFF+dr)**2)
    cchi[0]=2*cchi[1]-cchi[2]
    return cchi.imag
def kaiser_window():
    """Published implementation's beta=4 Kaiser support 1..19 Angstrom^-1."""
    k=KSTEP*np.arange(NFFT)
    arg=np.maximum(0.,1.-((k-10.)/9.)**2)
    return (np.i0(4*np.sqrt(arg))-1)/(np.i0(4)-1)
def transform(chi):
    """Negative-sign FFT; factor dk/sqrt(pi), zero padded to 2048."""
    k=KSTEP*np.arange(len(chi));buf=np.zeros(NFFT,dtype=complex)
    buf[:len(chi)]=chi*kaiser_window()[:len(chi)]*k**2
    return (KSTEP/np.sqrt(np.pi))*np.fft.fft(buf)[:NFFT//2]
def selected_transform(chi):
    lo=int(.01+1.4/RSTEP);hi=int(.01+3./RSTEP)
    return transform(chi)[lo:hi]
def epsilon_r():
    # Unit input weighting is a fit normalization, not an invented measured noise estimate.
    return 1./(2*np.sqrt(np.pi*5/(KSTEP*(17.**5-3.**5))))
def residuals(p,k,chi,splines):
    v=selected_transform(chi-forward(p,k,splines))/epsilon_r()
    return np.stack((v.real,v.imag),axis=-1).ravel()
def pair_density(p,r=R):
    """12-neighbor Gaussian radial density in Angstrom^-1, not a measured truth."""
    if p[2]<=0:raise ValueError('nonpositive fitted pair variance')
    return DEGENERACY*np.exp(-.5*(r-(REFF+p[3]))**2/p[2])/np.sqrt(2*np.pi*p[2])
def rfactor(p,k,chi,splines):
    diff=selected_transform(chi-forward(p,k,splines))
    ref=selected_transform(chi)
    return float(np.vdot(diff,diff).real/np.vdot(ref,ref).real)

import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((HERE/'metrics.json').read_text())
        for name in ('profile_nrmse','parameter_error','rfactor'):
            if not np.isfinite(cfg[name+'_boundary']) or cfg[name+'_boundary']<=0:raise ValueError('undefined evaluator boundary')
        reference_path=HERE/'ground_truth.npz' if (HERE/'ground_truth.npz').exists() else HERE/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as a:ref={k:a[k] for k in ('reconstruction','parameters')}
        for k,shape in [('reconstruction',(1,801)),('parameters',(1,4))]:
            if ref[k].shape!=shape or ref[k].dtype!=np.float64 or not np.isfinite(ref[k]).all():raise ValueError('malformed evaluator asset')
        if np.ptp(ref['reconstruction'])<=0:raise ValueError('degenerate evaluator reference')
        x={}
        with np.load(path,allow_pickle=False) as a:
            for k,shape in [('reconstruction',(1,801)),('parameters',(1,4))]:
                if k not in a:raise ValueError('missing required output '+k)
                x[k]=a[k]
                if x[k].shape!=shape or x[k].dtype!=np.float64 or not np.isfinite(x[k]).all():raise ValueError('invalid shape/dtype/finiteness: '+k)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:data={k:a[k][0] for k in a.files}
        p=x['parameters'][0];k=model_grid(data);chi=np.interp(k,data['k'],data['chi'])
        quality=rfactor(p,k,chi,scattering_splines(data)) if p[2]>0 and abs(REFF+p[3])>.1 else 1e100
        m=dict(profile_nrmse=float(np.sqrt(np.mean((x['reconstruction']-ref['reconstruction'])**2))/np.ptp(ref['reconstruction'])),parameter_error=float(np.max(np.abs(x['parameters']-ref['parameters'])/PARAMETER_SCALES)),rfactor=quality)
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('undefined metric')
        th={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in m}
        passed=all(m[k]<=th[k]['boundary'] for k in m)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=m,thresholds=th);return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
