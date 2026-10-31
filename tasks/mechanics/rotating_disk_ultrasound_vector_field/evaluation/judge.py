"""Standalone complete-field metrics and independently derived RF phase check.

No solver or submitted code is imported. Every observation used by the
physics metric is reconstructed from the public RF/grid/support NPZ.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy import signal,sparse,optimize,ndimage

HERE=Path(__file__).resolve().parent
EXPECTED_SHAPE=(1,2,251,251)

def load_prediction(path,role='prediction'):
    if not Path(path).is_file():raise ValueError(f'{role} file does not exist')
    with np.load(path,allow_pickle=False) as d:
        if set(d.files)!={'reconstruction'}:raise ValueError(f'{role} requires exactly reconstruction')
        a=d['reconstruction']
        if a.shape!=EXPECTED_SHAPE:raise ValueError(f'{role} expected shape {EXPECTED_SHAPE}, received {a.shape}')
        if a.dtype.kind!='f' or a.dtype.itemsize not in [4,8]:raise ValueError(f'{role} requires real float32 or float64')
        if not np.isfinite(a).all():raise ValueError(f'{role} contains non-finite values')
        return a[0].astype(float)

def field_metrics(estimate,reference):
    if np.ptp(reference)<=0 or np.linalg.norm(reference)==0:raise ValueError('Degenerate evaluator reference')
    n=np.linalg.norm(estimate)
    return {'ncc':float(estimate@reference/(n*np.linalg.norm(reference))) if n else 0.,
            'nrmse':float(np.sqrt(np.mean((estimate-reference)**2))/np.ptp(reference))}

def observation_phasors(path):
    """Derive both31-pair Hamming-averaged complex correlations from publicRF.

    Explicit interpolation, phase rotation, author parabolic virtual-array TX
    reader, angular aperture and instrument-directivity choice are implemented
    here independently of src. The same known geometry/constants are required
    to evaluate the declared measurement model.
    """
    with np.load(path,allow_pickle=False) as d:
        if set(d.files)!={'rf','observation_roi','x','z'}:raise ValueError('Evaluator public input keys invalid')
        rf=d['rf'];mask=d['observation_roi'];x=d['x'];z=d['z']
        if rf.shape!=(1,32,128,334) or rf.dtype!=np.int16:raise ValueError('Evaluator RF invalid')
        if mask.shape!=(1,251,251) or mask.dtype!=np.bool_:raise ValueError('Evaluator public support invalid')
        if x.shape!=(1,251) or z.shape!=(1,251) or not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Evaluator public coordinates invalid')
        rf=rf[0].transpose(2,1,0).astype(float);mask=mask[0]
    fs=6666666.666666667;fc=5e6;c=1480.;t0=9.95e-6;theta=np.pi/12
    time=t0+np.arange(334)/fs
    numerator,denominator=signal.butter(5,(fc*.15)/fs)
    mixed=rf*np.exp(-2j*np.pi*fc*time)[:,None,None]
    iq=2*signal.filtfilt(numerator,denominator,mixed,axis=0,padlen=15,padtype='odd')
    flat=iq.reshape(334*128,32,order='F')
    gx,gz=np.meshgrid(x[0],z[0]);gx=gx.ravel(order='F');gz=gz.ravel(order='F')
    elements=(np.arange(128)-63.5)*.000298
    wavelength=c/(fc*1.075)
    f=lambda a:abs(np.cos(a+theta)*np.sinc(.000262/wavelength*np.sin(a+theta))-.71)
    alpha=optimize.minimize_scalar(f,bounds=(0,np.pi/2-theta),method='bounded',options={'xatol':np.pi/100}).x
    aperture_ratio=.5/np.tan(alpha);correlations=[]
    for angle in [theta,-theta]:
        image=np.empty((len(gx),32),dtype=complex)
        for start in range(0,len(gx),1024):
            xx=gx[start:start+1024,None];zz=gz[start:start+1024,None]
            offset=xx-elements[None];distance=np.hypot(offset,zz);normal_distance=abs(offset)
            nearest=np.argmin(normal_distance,axis=1);rows=np.arange(len(xx));tx=distance[rows,nearest].copy()
            interior=(nearest>0)&(nearest<127);r=rows[interior];j=nearest[interior]
            dl=distance[r,j-1];dr=distance[r,j+1]
            nl=normal_distance[r,j-1];nm=normal_distance[r,j];nr=normal_distance[r,j+1]
            tx[interior]=(dl+dr)/2-(dr-dl)*(nr-nl)/(2*nl-4*nm+2*nr)
            travel=(tx[:,None]+distance)/c;clock=(travel-t0)*fs+1
            keep=(clock>=1)&(clock<=333)&(abs(offset-zz*np.tan(angle))<=zz/np.cos(angle)/2/aperture_ratio)
            row,element=np.nonzero(keep);sample=clock[row,element];lower=np.floor(sample).astype(np.int64)
            fraction=sample-lower;phase=np.exp(2j*np.pi*fc*travel[row,element])
            columns=lower-1+334*element
            operator=sparse.coo_matrix((np.r_[(1-fraction)*phase,fraction*phase],(np.r_[row,row],np.r_[columns,columns+1])),shape=(len(xx),334*128)).tocsc()
            image[start:start+len(xx)]=operator@flat
        image=image.reshape(251,251,32,order='F')
        correlation=np.sum(image[:,:,:-1]*np.conj(image[:,:,1:]),axis=2)
        window=np.outer(np.hamming(11),np.hamming(11))
        correlations.append(ndimage.convolve(correlation.real,window,mode='nearest')+1j*ndimage.convolve(correlation.imag,window,mode='nearest'))
    return np.stack(correlations),mask

def phasor_metric(estimate,correlations,mask):
    vx,vz=estimate;theta=np.pi/12
    projected=np.stack([-.5*(np.sin(theta)*vx+(1+np.cos(theta))*vz),.5*(np.sin(theta)*vx-(1+np.cos(theta))*vz)])
    observed=correlations[:,mask];weight=abs(observed)
    if np.any(weight==0) or not np.isfinite(weight).all():raise ValueError('Undefined observed phase')
    predicted=np.exp(-1j*np.pi*projected[:,mask]/.74)
    return float(np.sqrt(np.sum(weight*abs(predicted-observed/weight)**2)/weight.sum()))

def score_fields(estimate,reference,correlations,mask):
    values={}
    for i,key in enumerate(['vx','vz']):values.update({key+'_'+k:v for k,v in field_metrics(estimate[i,mask],reference[i,mask]).items()})
    values['weighted_phasor_rms']=phasor_metric(estimate,correlations,mask)
    values['outside_support_maxabs']=float(np.max(abs(estimate[:,~mask]))) if (~mask).any() else 0.
    return values

def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        root=Path(asset_dir);config=json.loads((root/'metrics.json').read_text(encoding='utf8'));limits=config['thresholds']
        expected={'vx_ncc','vx_nrmse','vz_ncc','vz_nrmse','weighted_phasor_rms','outside_support_maxabs'}
        if set(limits)!=expected:raise ValueError('Undefined evaluator metric configuration')
        for key,definition in limits.items():
            boundary=definition['boundary'];comparison=definition['comparison'];expected_comparison='>=' if key.endswith('_ncc') else '<='
            if comparison!=expected_comparison or not np.isfinite(boundary):raise ValueError('Invalid evaluator threshold')
            if key.endswith('_ncc') and not 0<=boundary<=1:raise ValueError('Invalid NCC boundary')
            if not key.endswith('_ncc') and boundary<0:raise ValueError('Negative error boundary')
            if key.endswith('_nrmse') and boundary<=0:raise ValueError('Numerical reference tolerance must be nonzero')
        truth=root/'ground_truth.npz';reference=load_prediction(truth if truth.exists() else root/'baseline_reference.npz','reference')
        estimate=load_prediction(prediction)
        correlations,mask=observation_phasors(root.parent/'data/raw_data.npz')
        values=score_fields(estimate,reference,correlations,mask)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined evaluation metric')
        passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in limits.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=limits)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True,type=Path);parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
