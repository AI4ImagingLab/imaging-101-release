"""Independent strict evaluator; never imports submitted or reference solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.constants import mu_0
ROOT=Path(__file__).resolve().parent
SHAPE=(1,2,301,401)

def cosine(x,y):
    a=np.linalg.norm(x);b=np.linalg.norm(y)
    if b==0:raise ValueError('Undefined zero reference norm')
    return float(np.vdot(x.ravel(),y.ravel()).real/(a*b)) if a else 0.

def relative_l2(x,y):
    b=np.linalg.norm(y)
    if b==0:raise ValueError('Undefined zero reference norm')
    return float(np.linalg.norm(x-y)/b)

def physical_metrics(current,data):
    field=np.pad(data['flux'][0],((75,75),(100,100)),mode='linear_ramp')
    x=data['xs'][0];y=data['ys'][0]
    xx=np.linspace(x[0]-100*(x[1]-x[0]),x[-1]+100*(x[1]-x[0]),401)
    yy=np.linspace(y[0]-75*(y[1]-y[0]),y[-1]+75*(y[1]-y[0]),301)
    kx,ky=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(401,xx[1]-xx[0]))*2*np.pi,np.fft.fftshift(np.fft.fftfreq(301,yy[1]-yy[0]))*2*np.pi)
    k=np.sqrt(kx*kx+ky*ky);mask=k>0
    p=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data['psf_coarse'][0])))
    jk=np.stack([np.fft.fftshift(np.fft.fft2(v)) for v in current])
    predicted=np.zeros(k.shape,dtype='complex128')
    predicted[mask]=(-.5j*mu_0)*np.exp(-k[mask])*p[mask]*(kx[mask]*jk[1,mask]-ky[mask]*jk[0,mask])/k[mask]
    window=np.where(k<=np.sqrt(2),.5*(1+np.cos(np.pi*k/np.sqrt(2))),0.)
    target=np.fft.fftshift(np.fft.fft2(field))*window;target[~mask]=0
    denom=np.linalg.norm(k*np.sqrt(abs(jk[0])**2+abs(jk[1])**2))
    return {'filtered_forward_residual':float(np.linalg.norm(predicted-target)/np.linalg.norm(target)),
            'divergence_ratio':float(np.linalg.norm(kx*jk[0]+ky*jk[1])/denom) if denom else 0.,
            'dc_ratio':float(np.linalg.norm(np.mean(current,axis=(-2,-1)))/max(np.sqrt(np.mean(current**2)),np.finfo(float).tiny))}

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        thresholds={name:{'comparison':comparison,'boundary':config[name+'_boundary']} for name,comparison in config['comparisons'].items()}
        required={'ncc_jx','relative_l2_jx','ncc_jy','relative_l2_jy','filtered_forward_residual','divergence_ratio','dc_ratio'}
        if set(thresholds)!=required:raise ValueError('Invalid required metric configuration')
        for name,item in thresholds.items():
            if item['comparison'] not in ['>=','<='] or type(item['boundary']) not in (int,float) or not np.isfinite(item['boundary']) or item['boundary']<=0:raise ValueError('Invalid threshold '+name)
        report['thresholds']=thresholds
        reference=ROOT/'ground_truth.npz'
        if not reference.exists():reference=ROOT/'baseline_reference.npz'
        with np.load(reference,allow_pickle=False) as f:y=f['reconstruction'].copy()
        with np.load(prediction,allow_pickle=False) as f:x=f['reconstruction'].copy()
        for a,role in [(x,'prediction'),(y,'reference')]:
            if a.shape!=SHAPE:raise ValueError(role+' must have shape '+str(SHAPE))
            if a.dtype!=np.dtype('float64'):raise ValueError(role+' must be real float64')
            if not np.isfinite(a).all():raise ValueError(role+' contains NaN/Inf')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            values={}
            for component,name in enumerate(['jx','jy']):values['ncc_'+name]=cosine(x[0,component],y[0,component]);values['relative_l2_'+name]=relative_l2(x[0,component],y[0,component])
            with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as f:data={key:f[key] for key in f.files}
            values.update(physical_metrics(x[0],data))
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite metric')
        passed=all(values[name]>=v['boundary'] if v['comparison']=='>=' else values[name]<=v['boundary'] for name,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report.update(metrics={},errors=[str(exc)]);return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
