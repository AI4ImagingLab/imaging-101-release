"""Independent evaluator for every iron-site parameter and transmitted spectrum."""
from pathlib import Path
import argparse,json,sys
import numpy as np
SHAPES={'reconstruction':(1,4,4,3),'optical_thickness':(1,4,4),'baseline':(1,4),'transmission':(1,4,512)}

def load(path):
    with np.load(path,allow_pickle=False) as f:
        result={}
        for k,shape in SHAPES.items():
            if k not in f:raise ValueError(f'Missing required key {k}')
            x=f[k]
            if x.shape!=shape:raise ValueError(f'{k} must have shape {shape}')
            if x.dtype!=np.float64:raise ValueError(f'{k} must be float64')
            if not np.all(np.isfinite(x)):raise ValueError(f'{k} contains nonfinite values')
            result[k]=x
    return result

def physical_spectra(velocity,sites,thickness,baseline):
    e=np.linspace(-.989,.989,64);a=np.array([.027181,22.7718,-33.6937])
    offsets=np.zeros(64);jac=np.zeros(64)
    for k in range(1,4):offsets+=a[k-1]*np.log((k+e)/(k-e));jac+=2*k*a[k-1]/(k*k-e*e)
    half=.098*.28/2;weights=half/np.pi/(offsets**2+half**2)*jac*(e[1]-e[0])
    answer=[]
    for v,component,intensity,level in zip(velocity,sites,thickness,baseline):
        depth=np.zeros((64,len(v)))
        for (delta,split,width),strength in zip(component,intensity):
            gamma=width*.28/2
            for center in [delta-split/2,delta+split/2]:
                deviation=.28*v[None]+offsets[:,None]-.28*center
                depth+=np.pi*half*strength*.5*(gamma/np.pi/(deviation**2+gamma**2))
        answer.append(level*(np.sum(weights[:,None]*np.exp(-depth),axis=0)+1-np.sum(weights)))
    return np.array(answer)

def score(pred,reference,velocity,observed,cfg):
    sites=pred['reconstruction'][0];thickness=pred['optical_thickness'][0]
    if np.any(sites[:,:,1:]<=0) or np.any(thickness<0):raise ValueError('Splittings/widths must be positive and thickness nonnegative')
    for spectrum in sites:
        order=np.lexsort((spectrum[:,1],spectrum[:,0]))
        if not np.array_equal(order,np.arange(4)):raise ValueError('Sites must follow canonical isomer-shift then splitting order')
    metrics={};thresholds={}
    for i,name in enumerate(['isomer_shift_mm_per_s','quadrupole_splitting_mm_per_s','FWHM_mm_per_s']):
        metrics[name]=float(np.max(abs(pred['reconstruction'][...,i]-reference['reconstruction'][...,i])))
        thresholds[name]={'comparison':'<=','boundary':cfg['site_parameter_absolute_boundary_mm_per_s']}
    for key,boundary in [('optical_thickness','optical_thickness_absolute_boundary'),('baseline','baseline_absolute_boundary'),('transmission','transmission_absolute_boundary')]:
        metrics[key+'_maximum_absolute_error']=float(np.max(abs(pred[key]-reference[key])))
        thresholds[key+'_maximum_absolute_error']={'comparison':'<=','boundary':cfg[boundary]}
    physical=physical_spectra(velocity,sites,thickness,pred['baseline'][0])
    metrics['parameter_spectrum_self_consistency']=float(np.max(abs(physical-pred['transmission'][0])))
    thresholds['parameter_spectrum_self_consistency']={'comparison':'<=','boundary':cfg['self_consistency_absolute_boundary']}
    for i in range(4):
        name=f'observed_transmission_RMS_sample_{i}'
        metrics[name]=float(np.sqrt(np.mean((pred['transmission'][0,i]-observed[i])**2)))
        thresholds[name]={'comparison':'<=','boundary':cfg['observed_rms_boundaries'][i]}
    if any(not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):raise ValueError('Invalid metric threshold')
    if any(not np.isfinite(value) for value in metrics.values()):raise ValueError('Undefined or overflowing evaluation metric')
    return metrics,thresholds

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        base=Path(__file__).resolve().parent;cfg=json.loads((base/'metrics.json').read_text(encoding='utf8'));truth=base/'ground_truth.npz'
        ref=load(truth if truth.exists() else base/'baseline_reference.npz');pred=load(Path(args.prediction))
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as data:v,y=data['velocity'][0],data['transmission'][0]
        metrics,thresholds=score(pred,ref,v,y,cfg);passed=all(metrics[k]<=r['boundary'] for k,r in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)]
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Report-write failure: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
