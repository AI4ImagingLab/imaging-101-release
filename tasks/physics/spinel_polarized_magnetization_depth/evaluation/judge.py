"""Standalone polarized-reflectivity evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import erf
SUBSTRATE_SLD=5.377

def unpack(parameters,dead=True):
    if dead:
        intensity,broadening,offset,rough_bottom,rough_top,dead_depth,magnetic,nuclear,thickness=parameters
    else:
        intensity,broadening,offset,rough_bottom,rough_top,magnetic,nuclear,thickness=parameters;dead_depth=0.
    return intensity,broadening,offset,rough_bottom,rough_top,dead_depth,magnetic,nuclear,thickness

def layer_stack(parameters,dead=True):
    _,_,_,sb,st,d,m,rho,h=unpack(parameters,dead)
    if dead:return np.array([0,h-d,d,0.]),np.array([0,rho,rho,SUBSTRATE_SLD]),np.array([0,m,0,0.]),np.array([st,sb,sb])
    return np.array([0,h,0.]),np.array([0,rho,SUBSTRATE_SLD]),np.array([0,m,0.]),np.array([st,sb])

def layer_reflectivity(q,width,nuclear,magnetic,roughness,spin):
    potential=(nuclear+spin*magnetic)*1e-6
    # Preserve original magnetic kernel's tiny positive absorption guard:
    # 1e-10 in the stored 1e-6 inverse-Angstrom-squared SLD convention.
    k=np.sqrt((np.asarray(q)[:,None]/2)**2-4*np.pi*potential[None,:]+4j*np.pi*1e-16)
    reflection=np.zeros(len(q),complex)
    for j in range(len(width)-2,-1,-1):
        a=k[:,j];b=k[:,j+1]
        interface=(a-b)/(a+b)*np.exp(-2*a*b*roughness[j]**2)
        phase=np.exp(2j*b*width[j+1]);reflection=(interface+reflection*phase)/(1+interface*reflection*phase)
    return abs(reflection)**2

def gaussian_resolution(q,values,sigma):
    # Exact integral of piecewise-linear intensity against a truncated Gaussian.
    # Whole intervals crossing the original exp(-6.907755...) cutoff are retained.
    q=np.asarray(q);values=np.asarray(values);sigma=np.asarray(sigma)
    limit=sigma*np.sqrt(2*6.90775527898213703123)
    left=np.clip(np.searchsorted(q,q-limit,side='right')-1,0,len(q)-2)
    right=np.clip(np.searchsorted(q,q+limit,side='left'),1,len(q)-1)
    x=q[None,:];center=q[:,None];s=sigma[:,None]
    errors=erf((x-center)/(np.sqrt(2)*s));gaussian=np.exp(-(x-center)**2/(2*s*s))
    slope=np.diff(values)/np.diff(q);intercept=values[1:]-slope*q[1:]
    area=.5*(slope[None,:]*center+intercept[None,:])*np.diff(errors,axis=1)-s/np.sqrt(2*np.pi)*slope[None,:]*np.diff(gaussian,axis=1)
    index=np.arange(len(q)-1)[None,:];mask=(index>=left[:,None])&(index<right[:,None])
    norm=errors[np.arange(len(q)),right]-errors[np.arange(len(q)),left]
    return 2*np.sum(np.where(mask,area,0),axis=1)/norm

def predictions(parameters,data,dead=True):
    intensity,broadening,offset,*_=unpack(parameters,dead);width,nuclear,magnetic,roughness=layer_stack(parameters,dead);result=[]
    for channel,spin in enumerate([1,-1]):
        angle=data['angle'][channel];wavelength=data['wavelength'][channel]
        q=4*np.pi/wavelength*np.sin(np.deg2rad(angle+offset))
        divergence=np.deg2rad(data['angle_resolution'][channel])/np.sqrt(8*np.log(2))
        extra=np.deg2rad(broadening)/np.sqrt(8*np.log(2))
        sigma=np.sqrt(data['q_resolution'][channel]**2+(4*np.pi/wavelength*np.cos(np.deg2rad(angle)))**2*(2*extra*divergence+extra**2))
        r=layer_reflectivity(q,width,nuclear,magnetic,roughness,spin)
        result.append(intensity*gaussian_resolution(q,r,sigma))
    return np.array(result)

def depth_profiles(parameters,depth,dead=True):
    _,_,_,sb,st,d,m,rho,h=unpack(parameters,dead)
    bottom=.5*(1+erf(depth/(np.sqrt(2)*sb)));top=.5*(1+erf((depth-h)/(np.sqrt(2)*st)))
    magnetic_bottom=.5*(1+erf((depth-d)/(np.sqrt(2)*sb)))
    nuclear=SUBSTRATE_SLD+(rho-SUBSTRATE_SLD)*bottom-rho*top
    magnetic=m*(magnetic_bottom-top)
    return np.stack([nuclear,magnetic],axis=-1)

LOWER=np.array([.98,0.,-.02,1.,1.,0.,0.,2.,60.])
UPPER=np.array([1.02,.08,.02,15.,15.,30.,2.,7.,180.])
def load_output(path):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in [('reconstruction',(1,521,2)),('parameters',(1,9))]:
            if key not in z:raise ValueError('Missing required key: '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape: '+key)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/64: '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite: '+key)
            out[key]=a
    p=out['parameters'][0]
    if np.any(p<LOWER) or np.any(p>UPPER):raise ValueError('Parameters outside physical contract bounds')
    if p[5]>=p[8]:raise ValueError('Dead layer cannot fill/exceed the film')
    if np.min(out['reconstruction']) < -1e-7:raise ValueError('Negative SLD profile in this nonnegative model')
    return out

def score(pred,ref,data):
    ranges=np.ptp(ref['reconstruction'][0],axis=0)
    if np.any(ranges<=0):raise ValueError('Undefined reference profile dynamic range')
    error=pred['reconstruction'][0]-ref['reconstruction'][0]
    p=pred['parameters'][0]
    forward=predictions(p,data);residual=(forward-data['reflectivity'])/data['uncertainty']
    consistency=pred['reconstruction'][0]-depth_profiles(p,data['depth'])
    return {'parameter_range_error':float(np.max(abs(p-ref['parameters'][0])/(UPPER-LOWER))),
      'nuclear_profile_nrmse':float(np.sqrt(np.mean(error[:,0]**2))/ranges[0]),
      'magnetic_profile_nrmse':float(np.sqrt(np.mean(error[:,1]**2))/ranges[1]),
      'profile_parameter_consistency':float(np.max(abs(consistency)/ranges)),
      'reduced_chi_square':float(np.sum(residual**2)/(168-9))}

def evaluate(prediction,evaluation_dir=None):
    folder=Path(evaluation_dir) if evaluation_dir else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        with np.load(folder.parent/'data/raw_data.npz',allow_pickle=False) as z:data={k:z[k][0] for k in z.files}
        path=folder/'ground_truth.npz'
        if not path.exists():path=folder/'baseline_reference.npz'
        try:
            ref=load_output(path)
            if np.max(abs(ref['reconstruction'][0]-depth_profiles(ref['parameters'][0],data['depth'])))>1e-5:raise ValueError('Reference profile does not match parameters')
        except Exception as exc:raise ValueError('Invalid hidden reference: '+str(exc)) from exc
        pred=load_output(prediction);values=score(pred,ref,data)
        for key,value in values.items():
            bound=config[key+'_boundary']
            if not np.isfinite(bound) or bound<=0 or not np.isfinite(value):raise ValueError('Invalid metric or boundary: '+key)
            report['thresholds'][key]={'comparison':'<=','boundary':bound}
        report['metrics']=values;report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in values.items());report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[str(exc)]);return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
