"""Standalone identifiable scattering/physical evaluation; never imports solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass,gaussian_filter,label
ROOT=Path(__file__).resolve().parent
SHAPE=(1,64,64,64)
VOXEL=3.75
SIDE=240.
DMAX=40.
NAMES=['chi2','mass_error','negative_mass_fraction','outside_mass_fraction','disconnected_fraction','porod_tail_relative_rms','rg_relative_error','reference_scattering_nrmse']

def scattering_samples(density: np.ndarray,data: dict) -> np.ndarray:
    """Complete complex FFT, independently binned, at every measured q."""
    axis=np.fft.fftfreq(64)*2*np.pi/VOXEL
    radius=np.sqrt(sum(a*a for a in np.meshgrid(axis,axis,axis,indexing='ij')))
    step=radius[radius>0].min()-1e-8;nb=int(radius.max()/step)
    edges=np.linspace(0,nb*step,nb+1);labels=np.searchsorted(edges,radius,'right')-1
    counts=np.bincount(labels.ravel())
    qcalc=np.bincount(labels.ravel(),weights=radius.ravel())/counts
    power=np.abs(np.fft.fftn(density*VOXEL**3))**2
    profile=np.bincount(labels.ravel(),weights=power.ravel())/counts
    q=data['q'][0,data['measured_mask'][0]];keep=qcalc<=q.max()
    return interp1d(qcalc[keep],profile[keep],kind='cubic',fill_value='extrapolate')(q)

def reference_scattering_nrmse(density: np.ndarray,reference: np.ndarray,data: dict) -> float:
    estimate=scattering_samples(density,data);target=scattering_samples(reference,data)
    scale=float(np.ptp(target))
    if not np.isfinite(scale) or scale<=0:raise ValueError('reference scattering dynamic range is undefined')
    return float(np.sqrt(np.mean((estimate-target)**2))/scale)

def physical_metrics(density: np.ndarray,data: dict) -> dict[str,float]:
    # Independent full FFT: every Cartesian frequency contributes exactly once.
    size=64;axis=np.fft.fftfreq(size)*2*np.pi/VOXEL
    qr=np.sqrt(sum(a*a for a in np.meshgrid(axis,axis,axis,indexing='ij')))
    step=qr[qr>0].min()-1e-8;nbins=int(qr.max()/step)
    bins=np.linspace(0,nbins*step,nbins+1);labels=np.searchsorted(bins,qr,'right')-1
    counts=np.bincount(labels.ravel());qcalc=np.bincount(labels.ravel(),weights=qr.ravel())/counts
    spectrum=np.fft.fftn(density*VOXEL**3)
    intensity=np.bincount(labels.ravel(),weights=np.abs(spectrum).ravel()**2)/counts
    q=data['q'][0];observed=data['intensity'][0];sigma=data['sigma'][0];fit=data['smoothed_intensity'][0];measured=data['measured_mask'][0]
    i0=fit[0];keep=qcalc<=q[measured].max()
    pred=interp1d(qcalc[keep],intensity[keep],kind='cubic',fill_value='extrapolate')(q[measured])
    chi=float(np.mean(((observed[measured]/i0-pred)/(sigma[measured]/i0))**2))
    target_inside=np.interp(qcalc[keep],q,fit)/i0
    tail=qcalc>q.max();tail_target=target_inside[-1]*(qcalc[tail]/qcalc[keep][-1])**-4
    tail_error=float(np.sqrt(np.mean((intensity[tail]/tail_target-1)**2)))
    mass=float(density.sum()*VOXEL**3)
    positive=np.clip(density,0,None);positive_mass=float(positive.sum())
    negative_fraction=float(-np.minimum(density,0).sum()/positive_mass) if positive_mass else 0.
    if positive_mass:
        shift=np.array(density.shape)//2-np.unravel_index(np.abs(density).argmax(),density.shape)
        centered=np.roll(density,shift,axis=(0,1,2))
        shift=np.rint(np.array(density.shape)//2-np.array(center_of_mass(np.abs(centered)))).astype(int)
        centered=np.roll(centered,shift,axis=(0,1,2));pos=np.clip(centered,0,None)
        x=(np.arange(size)-size//2)*VOXEL
        r=np.sqrt(sum(a*a for a in np.meshgrid(x,x,x,indexing='ij')))
        outside=float(pos[r>DMAX].sum()/positive_mass)
        blurred=gaussian_filter(pos,sigma=1.5,mode='wrap');labeled,_=label(blurred>=.2*blurred.max(),structure=np.ones((3,3,3)))
        weights=np.bincount(labeled.ravel(),weights=pos.ravel())[1:]
        disconnected=float(1-weights.max()/weights.sum()) if weights.size and weights.sum()>0 else 1.
    else:outside=1.;disconnected=1.
    guinier=measured&(q<=.08)&(observed>0)
    design=np.column_stack((np.ones(guinier.sum()),q[guinier]**2));weight=observed[guinier]/sigma[guinier]
    intercept,slope=np.linalg.lstsq(design*weight[:,None],np.log(observed[guinier])*weight,rcond=None)[0]
    if slope>=0:raise ValueError('measurement Guinier radius undefined')
    measurement_rg=np.sqrt(-3*slope)
    if 0<intensity[1]<intensity[0]:rg=np.sqrt(-3*np.log(intensity[1]/intensity[0]))/qcalc[1]
    else:rg=0.
    return {'chi2':chi,'mass_error':abs(mass-1),'negative_mass_fraction':negative_fraction,'outside_mass_fraction':outside,'disconnected_fraction':disconnected,'porod_tail_relative_rms':tail_error,'rg_relative_error':float(abs(rg-measurement_rg)/measurement_rg)}

def evaluate(prediction: str | Path) -> tuple[dict,int]:
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        with np.load(prediction,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('missing reconstruction')
            a=f['reconstruction']
        if a.shape!=SHAPE or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('prediction must be finite real float64 '+str(SHAPE))
        cfg=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        thresholds={}
        for name in NAMES:
            value=cfg[name+'_boundary']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError('invalid finite nonnegative threshold '+name)
            if name in ['negative_mass_fraction','outside_mass_fraction','disconnected_fraction'] and value>1:raise ValueError('fraction boundary exceeds one')
            thresholds[name]={'comparison':'<=','boundary':value}
        with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as f:data={k:f[k] for k in ['q','intensity','sigma','smoothed_intensity','measured_mask']}
        for key,v in data.items():
            if v.shape!=(1,485):raise ValueError('invalid public scattering calibration shape')
            if key=='measured_mask':
                if v.dtype!=np.bool_:raise ValueError('invalid observed mask')
            elif v.dtype!=np.float64 or not np.isfinite(v).all():raise ValueError('invalid public scattering data')
        if np.any(np.diff(data['q'][0])<=0) or np.any(data['sigma']<=0) or np.any(data['smoothed_intensity']<=0) or not data['measured_mask'].any():raise ValueError('invalid physical scattering measurements')
        truth=ROOT/'ground_truth.npz';reference_path=truth if truth.exists() else ROOT/'baseline_reference.npz'
        with np.load(reference_path,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('hidden reference missing reconstruction')
            reference=f['reconstruction']
        if reference.shape!=SHAPE or reference.dtype!=np.float64 or not np.isfinite(reference).all():raise ValueError('hidden reference must be finite real float64 '+str(SHAPE))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            metrics=physical_metrics(a[0],data)
            metrics['reference_scattering_nrmse']=reference_scattering_nrmse(a[0],reference[0],data)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite score')
        passed=all(metrics[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
