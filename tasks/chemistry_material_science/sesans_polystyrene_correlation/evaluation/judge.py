# Adapted explicit sphere/Hankel mathematics from SasView developers, BSD-3-Clause.
# Copyright2009-2026 SasView Developers; task LICENSE preserves redistribution terms.
import numpy as np
from scipy.special import j0
SLD=1.41
SLD_SOLVENT=2.70
THETA_MAX=.0168
RGRID=np.linspace(0.,50000.,501)
INITIAL=np.array([10000.,.0855])
BOUNDS=np.array([[100.,100000.],[.001,.5]])
PARAMETER_SCALES=np.array([10000.,.1])
NPRIOR=13
def hankel(data):
    """Build the exact upstream log-grid quadrature; no nominal length-error smearing."""
    z=data['spin_echo_length'];lam=data['wavelength']
    q_min=.1*2*np.pi/(len(z)*z[-1]);q_max=2*np.pi/(z[1]-z[0])
    q=np.exp(np.arange(np.log(q_min),np.log(q_max),np.log(1.0003)))
    dq=np.diff(q);dq=np.insert(dq,0,dq[0])
    H0=dq/(2*np.pi)*q
    H=np.outer(q,z);j0(H,out=H);H*=(dq*q/(2*np.pi)).reshape(-1,1)
    # Physical angular units. Upstream _make_sesans_transform passes a q cutoff
    # into an angle comparison; both versions are entirely inactive for this
    # bound dataset, verified against frozen H and the independent angle check.
    with np.errstate(invalid='ignore'):angle=np.arcsin(np.outer(q,lam/(2*np.pi)))
    H[~(angle<=THETA_MAX)]=0
    return q,H,H0
def sphere_amplitude(x):
    """Stable 3*j1(x)/x, preserving original float64 cutoff and four-term series."""
    x=np.asarray(x,dtype=float);out=np.empty_like(x);small=np.abs(x)<.1
    xx=x[small]*x[small]
    out[small]=1+xx*(-3./30.+xx*(3./840.+xx*(-3./45360.)))
    v=x[~small]
    out[~small]=3*(np.sin(v)/v-np.cos(v))/(v*v)
    return out
def sphere_intensity(p,q):
    """Macroscopic I(q) in cm^-1 for radius(Angstrom), volume fraction."""
    radius,phi=p;volume=(4*np.pi/3)*(radius*radius*radius)
    form=(SLD-SLD_SOLVENT)*volume*sphere_amplitude(q*radius)
    return phi*(1-phi)*(1e-4*form*form)/volume
def forward(p,q,H,H0):
    """Reduced depolarisation G(z)-G(0) in Angstrom^-2 cm^-1."""
    iq=sphere_intensity(p,q)
    return np.dot(H.T,iq)-np.dot(H0,iq)
def residuals(p,data,q,H,H0):
    return (forward(p,q,H,H0)-data['depolarisation'])/data['standard_error']
def bounded_residuals(p,data,q,H,H0):
    """Original Bumps projected-bound penalty and zero fixed/uniform prior residuals."""
    inside=np.clip(p,BOUNDS[:,0],BOUNDS[:,1]);stray=inside-p
    out=np.r_[residuals(inside,data,q,H,H0),np.zeros(NPRIOR)]
    cost=np.sum(stray*stray)
    if cost>0:cost+=1e6
    return out+np.sign(out)*(cost/len(out))
def autocovariance(p,r=RGRID):
    """Dimensionless spherical two-phase indicator covariance on radial separation."""
    radius,phi=p;u=r/radius
    return phi*(1-phi)*np.where(u<=2.,1-.75*u+u*u*u/16,0.)

import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((HERE/'metrics.json').read_text())
        for name in ('profile_nrmse','parameter_error','chisqr'):
            if not np.isfinite(cfg[name+'_boundary']) or cfg[name+'_boundary']<=0:raise ValueError('undefined evaluator metric')
        reference_path=HERE/'ground_truth.npz' if (HERE/'ground_truth.npz').exists() else HERE/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as a:ref={k:a[k] for k in ('reconstruction','parameters')}
        for k,shape in [('reconstruction',(1,501)),('parameters',(1,2))]:
            if ref[k].shape!=shape or ref[k].dtype!=np.float64 or not np.isfinite(ref[k]).all():raise ValueError('malformed evaluator reference')
        if np.ptp(ref['reconstruction'])<=0:raise ValueError('degenerate evaluator reference')
        x={}
        with np.load(path,allow_pickle=False) as a:
            for k,shape in [('reconstruction',(1,501)),('parameters',(1,2))]:
                if k not in a:raise ValueError('missing required output '+k)
                x[k]=a[k]
                if x[k].shape!=shape or x[k].dtype!=np.float64 or not np.isfinite(x[k]).all():raise ValueError('invalid shape/dtype/finiteness: '+k)
        p=x['parameters'][0]
        if np.any(p<BOUNDS[:,0]) or np.any(p>BOUNDS[:,1]):quality=1e100
        else:
            with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:data={k:a[k][0] for k in a.files}
            quality=float(np.sum(residuals(p,data,*hankel(data))**2))
        m=dict(profile_nrmse=float(np.sqrt(np.mean((x['reconstruction']-ref['reconstruction'])**2))/np.ptp(ref['reconstruction'])),parameter_error=float(np.max(np.abs(x['parameters']-ref['parameters'])/PARAMETER_SCALES)),chisqr=quality)
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('undefined metric')
        th={k:dict(comparison='<=',boundary=cfg[k+'_boundary']) for k in m};passed=all(m[k]<=th[k]['boundary'] for k in m)
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=m,thresholds=th);return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
