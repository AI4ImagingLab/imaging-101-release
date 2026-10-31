import argparse,json,sys
from pathlib import Path
import numpy as np

def positive_number(value):
    return type(value) in (int,float) and np.isfinite(value) and value>0


def model(p,f):
    mua=p[0];mus=p[1];transport=mua+mus;n=1.4;reff=-1.440/(n*n)+.710/n+.668+.0636*n;a=(1-reff)/(2*(1+reff));attenuation=np.sqrt(3*mua[None]*transport[None]+(2*np.pi*f[:,None,None])**2);ratio=attenuation/transport[None];return 3*a*(mus/transport)[None]/(ratio+1)/(ratio+3*a)

def demod(phase):
    a,b,c=phase.astype('float32');return (a+b+c)/np.float32(3),np.sqrt(np.float32(2)*((a-b)**2+(b-c)**2+(c-a)**2))/np.float32(3)

def observations(raw):
    hand=raw['hand_phase'][0];phantom=raw['phantom_phase'][0];rcal=raw['phantom_reflectance'][0];freq=raw['frequency_mm'][0]
    if hand.shape!=(5,3,500,400) or phantom.shape!=hand.shape or rcal.shape!=(5,) or freq.shape!=(5,):raise ValueError('Invalid evaluator calibration shapes')
    if not all(np.all(np.isfinite(v)) for v in [hand,phantom,rcal,freq]):raise ValueError('Nonfinite evaluator calibration')
    hs,ha=demod(hand[0]);ps,pa=demod(phantom[0]);cal=np.empty((500,400,5),dtype='float32');cal[:,:,0]=np.float32(rcal[0])*(hs/ps)
    for i in range(1,5):_,ha=demod(hand[i]);_,pa=demod(phantom[i]);cal[:,:,i]=np.float32(rcal[i])*(ha/pa)
    if not np.all(np.isfinite(cal)):raise ValueError('Invalid calibration denominator')
    return np.asfortranarray(cal).reshape(50,10,40,10,5).mean(3).mean(1).transpose(2,0,1),freq

def evaluate(prediction,root=None):
    root=Path(root) if root else Path(__file__).resolve().parent.parent;report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((root/'evaluation/metrics.json').read_text(encoding='utf-8'));factor=cfg['pixel_objective_ratio_boundary'];floor=cfg['pixel_objective_absolute_floor'];limit=cfg['forward_relative_l2_boundary']
        if not all(positive_number(v) for v in [factor,floor,limit]):raise ValueError('Invalid evaluator metric boundaries')
        with np.load(root/'data/raw_data.npz',allow_pickle=False) as raw:obs,freq=observations(raw)
        truth=(root/'evaluation/ground_truth.npz').is_file();refpath=root/'evaluation/ground_truth.npz' if truth else root/'evaluation/baseline_reference.npz'
        with np.load(refpath,allow_pickle=False) as a:
            ref={}
            for name,shape in [('reconstruction',(1,2,50,40)),('reflectance',(1,5,50,40))]:
                x=a[name]
                if x.shape!=shape or x.dtype!=np.dtype('float64') or not np.all(np.isfinite(x)):raise ValueError('Invalid evaluator reference '+name)
                ref[name]=x.copy()
        if np.any(ref['reconstruction']<=0):raise ValueError('Nonphysical evaluator reference')
        with np.load(prediction,allow_pickle=False) as a:
            out={}
            for name,shape in [('reconstruction',(1,2,50,40)),('reflectance',(1,5,50,40))]:
                if name not in a:raise ValueError('Missing output '+name)
                x=a[name]
                if x.shape!=shape:raise ValueError('Invalid shape '+name)
                if x.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 '+name)
                if not np.all(np.isfinite(x)):raise ValueError('Nonfinite '+name)
                out[name]=x.copy()
        if np.any(out['reconstruction']<=0):raise ValueError('Nonpositive optical coefficient')
        expected=model(out['reconstruction'][0],freq);reference_model=model(ref['reconstruction'][0],freq)
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(reference_model)):raise ValueError('Nonfinite physical forward result')
        if np.linalg.norm(ref['reflectance'][0]-reference_model)/np.linalg.norm(reference_model)>limit:raise ValueError('Inconsistent evaluator reference fields')
        per_pixel=np.sum((expected[:2]-obs[:2])**2,axis=0);reference_sse=np.sum((reference_model[:2]-obs[:2])**2,axis=0)
        report['metrics']={'maximum_pixel_bound_fraction':float(np.max(per_pixel/(factor*reference_sse+floor))),'forward_relative_l2':float(np.linalg.norm(out['reflectance'][0]-expected)/np.linalg.norm(expected))}
        report['thresholds']={'maximum_pixel_bound_fraction':dict(comparison='<=',boundary=1.),'forward_relative_l2':dict(comparison='<=',boundary=float(limit))}
        if truth:
            bound=cfg.get('truth_relative_l2_boundary')
            if not positive_number(bound):raise ValueError('Actual truth requires explicit truth_relative_l2_boundary')
            for name in out:
                key='truth_relative_l2_'+name;report['metrics'][key]=float(np.linalg.norm(out[name]-ref[name])/np.linalg.norm(ref[name]));report['thresholds'][key]=dict(comparison='<=',boundary=float(bound))
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite metric')
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items());report['status']='pass' if report['passed'] else 'fail';return report,0 if report['passed'] else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)]);return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
