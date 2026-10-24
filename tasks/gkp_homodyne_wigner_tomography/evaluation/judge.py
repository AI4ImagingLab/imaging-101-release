"""Independent standalone evaluator, no submitted/reference solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import eval_genlaguerre,gammaln

BASE=Path(__file__).resolve().parent
SHAPES={'density_matrix':(1,31,31),'wigner':(1,201,201)}
METRICS={
 'rho_relative_l2':('rho_relative_l2_boundary','less_equal'),
 'wigner_nrmse':('wigner_nrmse_boundary','less_equal'),
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
        if np.linalg.norm(reference['density_matrix'])==0 or np.ptp(reference['wigner'])==0:raise ValueError('Degenerate evaluator reference')
        ref=reference['density_matrix']
        if abs(np.trace(ref)-1)>1e-8 or np.max(abs(ref-ref.conj().T))>1e-8 or np.linalg.eigvalsh((ref+ref.conj().T)/2).min()<-1e-8:
            raise ValueError('Evaluator method reference is not a normalized physical density')
        prediction=load_arrays(prediction_path)
        values=score(prediction,reference)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite evaluation metric')
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
