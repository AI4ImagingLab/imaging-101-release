"""Independent full-cell evaluator with declared periodic origin ambiguity."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import maximum_filter

BASE=Path(__file__).resolve().parent
SHAPE=(1,72,24,96)

def load_density(path,role):
    if not Path(path).is_file():raise ValueError(role+' file does not exist')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError(role+' requires exactly reconstruction')
        x=z['reconstruction']
        if x.shape!=SHAPE:raise ValueError(role+' expected shape '+str(SHAPE))
        if x.dtype not in [np.dtype('float32'),np.dtype('float64')]:raise ValueError(role+' requires real float32 or float64')
        if not np.isfinite(x).all():raise ValueError(role+' has NaN or Infinity')
        return x[0].astype(np.float64)

def align_periodic(estimate,reference):
    """All-voxel translation/inversion only; no gain, crop or data-dependent mask.

    A real periodic trigonometric interpolant defines fractional shifts. The
    reference has no Nyquist support. The real part below evaluates that declared
    real interpolant; it never coerces a submitted complex array to real.
    """
    norm=float(np.linalg.norm(estimate)*np.linalg.norm(reference));N=estimate.size
    if norm==0:return estimate.copy(),np.zeros(3),False
    G=np.fft.fftn(reference);freq=[-2j*np.pi*np.fft.fftfreq(n) for n in estimate.shape]
    best=None
    for inverted in [False,True]:
        x=estimate[np.ix_(*[(-np.arange(n))%n for n in estimate.shape])] if inverted else estimate
        F=np.fft.fftn(x);corr=np.fft.ifftn(G*F.conj()).real
        peaks=np.flatnonzero(corr==maximum_filter(corr,size=3,mode='wrap'))
        peaks=peaks[np.argsort(corr.ravel()[peaks])[-8:]]
        starts=[np.zeros(3)]+[np.array(np.unravel_index(p,corr.shape),dtype=float) for p in peaks]
        C=G.conj()*F
        def objective(t):
            ex,ey,ez=[np.exp(k*s) for k,s in zip(freq,t)]
            cz=np.tensordot(C,ez,axes=(2,0));cy=cz@ey
            value=ex@cy
            grad=np.array([(ex*freq[0])@cy,ex@(cz@(ey*freq[1])),ex@(np.tensordot(C,ez*freq[2],axes=(2,0))@ey)])
            return -float(value.real/(norm*N)),-grad.real/(norm*N)
        for start in starts:
            fit=minimize(objective,start,jac=True,method='L-BFGS-B',bounds=[(v-1,v+1) for v in start],options={'ftol':1e-15,'gtol':1e-10,'maxiter':80})
            phase=np.exp(freq[0]*fit.x[0])[:,None,None]*np.exp(freq[1]*fit.x[1])[None,:,None]*np.exp(freq[2]*fit.x[2])[None,None,:]
            aligned=np.fft.ifftn(F*phase).real
            score=float(np.sum(aligned*reference)/(np.linalg.norm(aligned)*np.linalg.norm(reference)))
            if best is None or score>best[0]:best=(score,aligned,fit.x,inverted)
    return best[1],best[2],best[3]

def physical_metrics(density,raw_path):
    with np.load(raw_path,allow_pickle=False) as z:
        if set(z.files)!={'hkl','intensity','sigma_intensity'}:raise ValueError('Invalid evaluator public observation keys')
        h=z['hkl'];I=z['intensity'];sig=z['sigma_intensity']
        if h.shape!=(1,4508,3) or I.shape!=(1,4508) or sig.shape!=I.shape:raise ValueError('Invalid evaluator public shapes')
        if h.dtype.kind not in 'iu' or not all(np.isfinite(v).all() for v in [h,I,sig]):raise ValueError('Invalid evaluator observation data')
        h=h[0];I=I[0]
    norm=np.linalg.norm(I)
    if norm<=0:raise ValueError('Degenerate evaluator observations')
    F=np.fft.ifftn(10*density)
    amplitudes=abs(F[tuple((h%np.array(density.shape)).T)])
    residual=float(np.linalg.norm(amplitudes**2-I)/norm)
    rms=float(np.sqrt(np.mean(density*density)))
    return {'intensity_relative_l2':residual,'mean_relative':float(abs(density.mean())/rms) if rms>0 else 0.}

def evaluate(prediction,asset_dir=BASE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        directory=Path(asset_dir);config=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        thresholds={}
        for name,comparison in [('ncc','>='),('nrmse','<='),('intensity_relative_l2','<='),('mean_relative','<=')]:
            value=config[name+'_boundary']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Invalid evaluator boundary '+name)
            if name=='ncc' and not 0<=value<=1:raise ValueError('Invalid NCC boundary')
            if name!='ncc' and value<=0:raise ValueError('Error boundaries require measured nonzero tolerance')
            thresholds[name]={'comparison':comparison,'boundary':value}
        truth=directory/'ground_truth.npz'
        reference=load_density(truth if truth.exists() else directory/'baseline_reference.npz','reference')
        if np.ptp(reference)<=0 or np.linalg.norm(reference)==0:raise ValueError('Degenerate evaluator reference')
        estimate=load_density(prediction,'prediction')
        aligned,shift,inverted=align_periodic(estimate,reference)
        den=np.linalg.norm(aligned)*np.linalg.norm(reference)
        metrics={'ncc':float(np.sum(aligned*reference)/den) if den>0 else 0.,'nrmse':float(np.sqrt(np.mean((aligned-reference)**2))/np.ptp(reference))}
        metrics.update(physical_metrics(estimate,directory.parent/'data/raw_data.npz'))
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined evaluation metric')
        passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds,alignment={'periodic_shift_voxels':shift.tolist(),'spatial_inversion':inverted})
        return report,0 if passed else 1
    except Exception as e:
        report['errors']=[f'{type(e).__name__}: {e}'];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True,type=Path);parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Cannot write evaluation report: '+str(e),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
