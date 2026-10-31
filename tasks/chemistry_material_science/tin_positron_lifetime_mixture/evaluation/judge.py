import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr,log_ndtr
SHAPES={'reconstruction':(1,6,3),'intensity':(1,6,3),'irf_fwhm_ps':(1,6,2),'irf_shift_ps':(1,6,2),'irf_weight':(1,6,2),'background':(1,6),'prediction':(1,6,1000)}

def positive_number(value):
    return type(value) in (int,float) and np.isfinite(value) and value>0


def read(path):
    with np.load(path,allow_pickle=False) as a:
        out={}
        for key,shape in SHAPES.items():
            if key not in a:raise ValueError('Missing output '+key)
            v=a[key]
            if v.shape!=shape:raise ValueError('Invalid shape '+key)
            if v.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 '+key)
            if not np.all(np.isfinite(v)):raise ValueError('Nonfinite '+key)
            out[key]=v[0].copy()
    if np.any(out['reconstruction']<=0) or np.any(out['irf_fwhm_ps']<=0):raise ValueError('Nonpositive lifetime or IRF width')
    if any(np.any(out[k]<0) for k in ['intensity','irf_weight','background']):raise ValueError('Negative component weight or background')
    if np.any(np.diff(out['reconstruction'],axis=1)<0) or np.any(np.diff(out['irf_fwhm_ps'],axis=1)<0):raise ValueError('Components must follow declared lifetime/width order')
    return out

def model(out,area):
    t=(np.arange(1001,dtype=float)-33)*50.;curves=[]
    for i in range(6):
        cdf=np.zeros(1001)
        for width,mean,weight in zip(out['irf_fwhm_ps'][i],out['irf_shift_ps'][i],out['irf_weight'][i]):
            sigma=width/(2*np.sqrt(2*np.log(2)));delta=t-mean;z=delta/sigma
            for tau,intensity in zip(out['reconstruction'][i],out['intensity'][i]):
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    term=np.exp(-delta/tau+.5*(sigma/tau)**2+log_ndtr(z-sigma/tau));cdf+=weight*intensity*(ndtr(z)-term)
        curves.append((area[i]-916*out['background'][i])*np.diff(cdf)+out['background'][i])
    return np.array(curves)

def evaluate(prediction,root=None):
    root=Path(root) if root else Path(__file__).resolve().parent.parent;report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((root/'evaluation/metrics.json').read_text(encoding='utf8'));raw_bounds=cfg['objective_boundary']
        if type(raw_bounds) is not list or len(raw_bounds)!=6 or not all(positive_number(v) for v in raw_bounds):raise ValueError('Invalid objective boundaries')
        bounds=np.asarray(raw_bounds);limit=cfg['forward_observation_normalized_l2_boundary']
        if bounds.shape!=(6,) or not np.all(np.isfinite(bounds)) or np.any(bounds<=0) or not positive_number(limit):raise ValueError('Invalid metric configuration')
        with np.load(root/'data/raw_data.npz',allow_pickle=False) as a:counts=a['counts'][0]
        if counts.shape!=(6,10000) or np.any(counts<0) or not np.all(np.isfinite(counts)):raise ValueError('Invalid evaluator measurements')
        y=counts.reshape(6,1000,10).sum(2).astype(float);area=y[:,33:949].sum(1);scale=np.linalg.norm(y,axis=1)
        if np.any(scale<=0):raise ValueError('Degenerate observation scale')
        truth=(root/'evaluation/ground_truth.npz').is_file();reference=read(root/('evaluation/ground_truth.npz' if truth else 'evaluation/baseline_reference.npz'));out=read(prediction);expected=model(out,area);refmodel=model(reference,area)
        if np.any(area-916*out['background']<=0) or np.any(area-916*reference['background']<=0):raise ValueError('Nonpositive signal area')
        if np.any(out['prediction']<0):raise ValueError('Negative predicted counts')
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(refmodel)):raise ValueError('Nonfinite forward calculation')
        if np.any(np.linalg.norm(reference['prediction']-refmodel,axis=1)/scale>limit):raise ValueError('Inconsistent evaluator reference')
        with np.errstate(over='ignore',invalid='ignore'):
            objective=np.sum((y[:,33:947]-expected[:,33:947])**2/(y[:,33:947]+1),axis=1)+1e8*(out['irf_weight'].sum(1)-1)**2
            consistency=np.linalg.norm(out['prediction']-expected,axis=1)/scale
        for i in range(6):
            for key,value,bound in [('objective',objective[i],bounds[i]),('forward_observation_normalized_l2',consistency[i],limit)]:
                name=f'{key}_{i}';report['metrics'][name]=float(value);report['thresholds'][name]=dict(comparison='<=',boundary=float(bound))
        if truth:
            bound=cfg.get('truth_relative_l2_boundary')
            if not positive_number(bound):raise ValueError('Actual truth requires explicit truth_relative_l2_boundary')
            for key in out:
                norm=np.linalg.norm(reference[key])
                if norm==0:raise ValueError('Zero truth requires explicit alternative metric')
                name='truth_relative_l2_'+key;report['metrics'][name]=float(np.linalg.norm(out[key]-reference[key])/norm);report['thresholds'][name]=dict(comparison='<=',boundary=float(bound))
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite metric')
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items());report['status']='pass' if report['passed'] else 'fail';return report,0 if report['passed'] else 1
    except Exception as exc:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)]);return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args();report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
