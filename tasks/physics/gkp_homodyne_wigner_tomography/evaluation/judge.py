"""Independent standalone evaluator, no submitted/reference solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import eval_genlaguerre,gammaln

BASE=Path(__file__).resolve().parent
SHAPES={'density_matrix':(1,31,31),'wigner':(1,201,201)}
METRICS={
 'mean_loglikelihood_deficit':('mean_loglikelihood_deficit_boundary','less_equal'),
 'rho_hermitian_relative':('rho_hermitian_relative_boundary','less_equal'),
 'rho_trace_error':('rho_trace_error_boundary','less_equal'),
 'rho_minimum_eigenvalue':('rho_minimum_eigenvalue_boundary','greater_equal'),
 'wigner_density_consistency_maxabs':('wigner_density_consistency_maxabs_boundary','less_equal')}

def load_arrays(path):
    if not Path(path).is_file():raise ValueError('NPZ file does not exist: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(SHAPES):raise ValueError('Required keys are exactly density_matrix and wigner')
        result={}
        for name,shape in SHAPES.items():
            a=z[name]
            if a.shape!=shape:raise ValueError(f'{name}: expected shape {shape}, got {a.shape}')
            allowed=(np.dtype('complex64'),np.dtype('complex128')) if name=='density_matrix' else (np.dtype('float32'),np.dtype('float64'))
            if a.dtype not in allowed:raise ValueError(f'{name}: invalid declared numeric dtype {a.dtype}')
            if not np.isfinite(a).all():raise ValueError(name+': values must be finite')
            result[name]=a[0].copy()
    return result

def independent_wigner(density):
    """Closed Laguerre basis; independent of the source's coupled recurrence."""
    axis=np.linspace(-7,7,201);Q,P=np.meshgrid(axis,axis)
    radius=Q*Q+P*P;beta=np.sqrt(2)*(Q-1j*P);gaussian=np.exp(-radius)/np.pi
    W=np.zeros_like(Q)
    for n in range(31):
        W+=density[n,n].real*(-1)**n*eval_genlaguerre(n,0,2*radius)*gaussian
        for d in range(1,31-n):
            basis=(-1)**n*np.exp((gammaln(n+1)-gammaln(n+d+1))/2)*beta**d*eval_genlaguerre(n,d,2*radius)*gaussian
            W+=2*np.real(density[n+d,n]*basis)
    return W

def score(prediction,reference):
    rho,W=prediction['density_matrix'],prediction['wigner']
    ref,refW=reference['density_matrix'],reference['wigner']
    norm=float(np.linalg.norm(ref));span=float(np.ptp(refW))
    if norm<=0 or span<=0:raise ValueError('Degenerate evaluator reference')
    hermitian=(rho+rho.conj().T)/2
    rho_norm=float(np.linalg.norm(rho))
    return {'rho_relative_l2':float(np.linalg.norm(rho-ref)/norm),
            'wigner_nrmse':float(np.sqrt(np.mean((W-refW)**2))/span),
            'rho_hermitian_relative':float(np.linalg.norm(rho-rho.conj().T)/max(rho_norm,1e-300)),
            'rho_trace_error':float(abs(np.trace(rho)-1)),
            'rho_minimum_eigenvalue':float(np.linalg.eigvalsh(hermitian).min()),
            'wigner_density_consistency_maxabs':float(np.max(abs(W-independent_wigner(rho))))}

def conditional_samples():
    """Observation-only variance calibration and public photon-confidence selection."""
    from scipy.optimize import curve_fit
    with np.load(BASE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        data={k:z[k][0] for k in z.files}
    def variance(angle,offset,amplitude,phase,low,high):
        theta=angle-offset+amplitude*np.sin(angle-phase)
        return low*np.cos(theta)**2+high*np.sin(theta)**2
    q=np.empty(data['adc'].size);angle=np.empty_like(q)
    for f in range(100):
        x=data['variance_phase'][f];y=.5*data['signal_variance'][f]/data['vacuum_variance'][f]
        fit,_=curve_fit(variance,x,y,p0=[((x[np.argmin(y)]+np.pi/2)%np.pi)-np.pi/2,0.,0.,y.min(),y.max()])
        mask=data['file_index']==f
        q[mask]=(data['adc'][mask]-data['quadrature_mean'][f])/np.sqrt(2*np.mean(data['vacuum_variance'][f]))
        raw=data['phase_index'][mask]*data['phase_step'][f]
        angle[mask]=raw-fit[0]+fit[1]*np.sin(raw-fit[2])
    cutoffs=np.array([5554.373134328358,5811.4427860696524,7281.3880597014922])
    keep=np.all(data['pnr_distance'].astype(np.float64)<cutoffs,axis=1)
    if not keep.any():raise ValueError('Empty conditional homodyne observation set')
    return q[keep],angle[keep]


def mean_loglikelihoods(densities,q,angle):
    """Chunked oscillator recurrence and direct Born probability, all accepted shots."""
    totals=np.zeros(len(densities))
    for first in range(0,q.size,4096):
        x=q[first:first+4096];phi=angle[first:first+4096]
        h=np.empty((31,x.size));h[0]=np.pi**(-.25)*np.exp(-x*x/2)
        h[1]=np.sqrt(2)*x*h[0]
        for n in range(1,30):h[n+1]=np.sqrt(2/(n+1))*x*h[n]-np.sqrt(n/(n+1))*h[n-1]
        amplitude=h*np.exp(1j*np.arange(31)[:,None]*phi)
        for index,rho in enumerate(densities):
            probability=np.sum(amplitude.conj()*(rho@amplitude),axis=0).real
            if np.any(probability<=0) or not np.isfinite(probability).all():totals[index]=-1e100
            elif totals[index]>-1e99:totals[index]+=np.log(probability).sum()
    return totals/q.size


def evaluate(prediction_path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((BASE/'metrics.json').read_text(encoding='utf-8'))
        thresholds={}
        for name,(key,direction) in METRICS.items():
            value=config.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Invalid evaluator threshold '+key)
            if key!='rho_minimum_eigenvalue_boundary' and value<0:raise ValueError('Negative evaluator error threshold '+key)
            thresholds[name]={'comparison':direction,'boundary':value}
        truth_path=BASE/'ground_truth.npz'
        reference=load_arrays(truth_path if truth_path.exists() else BASE/'baseline_reference.npz')
        if truth_path.exists():
            for metric in ('rho_relative_l2','wigner_nrmse'):
                boundary=config['actual_truth_'+metric+'_boundary']
                if isinstance(boundary,bool) or not np.isfinite(boundary) or boundary<0:raise ValueError('Invalid actual truth boundary')
                thresholds[metric]={'comparison':'less_equal','boundary':boundary}
        if np.linalg.norm(reference['density_matrix'])==0 or np.ptp(reference['wigner'])==0:raise ValueError('Degenerate evaluator reference')
        ref=reference['density_matrix']
        if abs(np.trace(ref)-1)>1e-8 or np.max(abs(ref-ref.conj().T))>1e-8 or np.linalg.eigvalsh((ref+ref.conj().T)/2).min()<-1e-8:
            raise ValueError('Evaluator method reference is not a normalized physical density')
        prediction=load_arrays(prediction_path)
        values=score(prediction,reference)
        q,angle=conditional_samples()
        candidate,standard=mean_loglikelihoods([prediction['density_matrix'],ref],q,angle)
        values['mean_loglikelihood_deficit']=float(max(0.,standard-candidate))
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite evaluation metric')
        values={name:values[name] for name in thresholds}
        passed=all(values[k]<=t['boundary'] if t['comparison']=='less_equal' else values[k]>=t['boundary'] for k,t in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as error:
        report['errors']=[str(error)]
        return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as error:
        print('Cannot write evaluation report: '+str(error),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
