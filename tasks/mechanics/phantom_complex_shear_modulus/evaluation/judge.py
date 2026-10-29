"""Standalone complex-field evaluator. Imports no solver or submitted code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.ndimage import convolve,median_filter
SHAPE=(1,2,25,80,128)
def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();span=np.ptp(b);norm=np.linalg.norm(b)
    if span<=0 or norm<=0:raise ValueError("Degenerate component reference")
    return dict(ncc=float(a@b/(np.linalg.norm(a)*norm)) if np.linalg.norm(a)>0 else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/span))

def observation_residual(volume,raw):
    """Independent explicit public-observation consistency for selected planar model."""
    coords=np.stack(np.meshgrid(np.arange(-4,5),np.arange(-4,5)),axis=-1).reshape(-1,2)
    powers=np.stack(np.meshgrid(np.arange(3),np.arange(3)),axis=-1).reshape(-1,2);powers=powers[powers.sum(1)<=2]
    matrix=np.array([[np.prod(c**p) for p in powers] for c in coords],float)
    kernels=np.linalg.pinv(matrix);kernels*=np.prod(np.where(powers==2,2,1),axis=1)[:,None]
    kernels={tuple(p):k.reshape(9,9,1,1) for p,k in zip(powers,kernels)}
    dx=np.diff(raw['x'][0]).mean();dy=np.diff(raw['y'][0]).mean();support=raw['support'][0].transpose(2,1,0).astype(bool)
    footprint=(np.outer([1,2,3,2,1],[1,2,3,2,1])>1)[:,:,None,None]
    def real_filter(a):
        med=median_filter(a,footprint=footprint,mode='reflect');diff=(a-med)/med
        return np.where(diff < -3,-3*med,np.where(diff>3,3*med,a))
    residual=[]
    for k,f in enumerate(raw['frequency'][0]):
        u=raw['wave'][0,k].transpose(3,2,1,0);u=convolve(u,kernels[(0,0)],mode='reflect')
        lap=convolve(u,kernels[(2,0)],mode='reflect')/dx**2+convolve(u,kernels[(0,2)],mode='reflect')/dy**2
        inv=1/lap;lap=1/(real_filter(abs(inv))*np.exp(1j*real_filter(np.angle(inv))))
        force=1000*(2*np.pi*f)**2*u;mu=volume[0,k].transpose(2,1,0)
        residual.append(float(np.linalg.norm((mu[...,None]*lap+force)[support])/np.linalg.norm(force[support])))
    return residual

def evaluate(prediction,asset_dir=None):
    base=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((base/'metrics.json').read_text(encoding='utf-8'))
        for key in ['ncc_boundary','nrmse_boundary','relative_helmholtz_residual_boundary']:
            if not isinstance(config.get(key),(int,float)) or not np.isfinite(config[key]):raise ValueError('Invalid metric config '+key)
        if not 0<config['ncc_boundary']<=1 or config['nrmse_boundary']<=0 or config['relative_helmholtz_residual_boundary']<=0:raise ValueError('Invalid metric boundaries')
        reference_path=base/('ground_truth.npz' if (base/'ground_truth.npz').is_file() else 'baseline_reference.npz')
        with np.load(reference_path,allow_pickle=False) as z:
            if set(z.files)!={'reconstruction'}:raise ValueError('Invalid reference keys')
            reference=z['reconstruction']
        if reference.shape!=SHAPE or reference.dtype!=np.complex128 or not np.isfinite(reference).all():raise ValueError('Invalid reference contract')
        with np.load(Path(__file__).resolve().parent.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        support=raw['support'][0].astype(bool)
        with np.load(Path(prediction),allow_pickle=False) as z:
            if set(z.files)!={'reconstruction'}:raise ValueError('Required sole key: reconstruction')
            volume=z['reconstruction']
        if volume.shape!=SHAPE:raise ValueError('Wrong reconstruction shape')
        if volume.dtype!=np.complex128:raise ValueError('Expected complex128 preserving storage and loss')
        if not np.isfinite(volume).all():raise ValueError('Nonfinite reconstruction')
        if np.any(volume[0,:,~support]!=0):raise ValueError('Exterior must be exactly zero')
        measurements={};thresholds={}
        for k in range(2):
            for component,extract in [('storage',np.real),('loss',np.imag)]:
                m=metrics(extract(volume[0,k][support]),extract(reference[0,k][support]))
                for metric,value in m.items():
                    key=f'{component}_{k}_{metric}';measurements[key]=value;thresholds[key]=dict(comparison='>=' if metric=='ncc' else '<=',boundary=config[metric+'_boundary'])
        residual=observation_residual(volume,raw)
        for k,value in enumerate(residual):measurements[f'helmholtz_{k}']=value;thresholds[f'helmholtz_{k}']=dict(comparison='<=',boundary=config['relative_helmholtz_residual_boundary'])
        if not all(np.isfinite(v) for v in measurements.values()):raise ValueError('Nonfinite metrics')
        passed=all(value>=thresholds[key]['boundary'] if thresholds[key]['comparison']=='>=' else value<=thresholds[key]['boundary'] for key,value in measurements.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measurements,thresholds=thresholds)
    except Exception as e:report['errors']=[str(e) or type(e).__name__]
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
