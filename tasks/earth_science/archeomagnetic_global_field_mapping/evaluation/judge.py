"""Standalone deterministic evaluator; no solver imports or execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import lpmv
import math
HERE=Path(__file__).resolve().parent
PUBLIC_DATA=HERE.parent/'data/raw_data.npz'
SCHEMA={'reconstruction':(1,5,3,36,72),'coefficients':(1,43,15)}

def validate_config(config):
    required=['ncc_boundary','nrmse_boundary']+[f'{name}_whitened_rms_boundary' for name in ('intensity','inclination','declination')]
    for key in required:
        if key not in config: raise ValueError(f'Missing evaluator threshold: {key}')
        value=config[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'Evaluator threshold must be finite numeric: {key}')
        if key=='ncc_boundary' and not -1<=value<=1: raise ValueError('NCC boundary must lie in [-1,1]')
        if key!='ncc_boundary' and value<0: raise ValueError(f'Error boundary must be nonnegative: {key}')

def field_at(coefficients,year,latitude,longitude):
    latitude=np.deg2rad(latitude); longitude=np.deg2rad(longitude)
    major,minor=6378137.,6356752.
    eccentricity=1-(minor/major)**2
    prime=major/np.sqrt(1-eccentricity*np.sin(latitude)**2)
    horizontal=prime*np.cos(latitude)
    vertical=prime*(1-eccentricity)*np.sin(latitude)
    radius=np.hypot(horizontal,vertical)/1000
    geocentric=np.arctan2(vertical,horizontal)
    theta=np.pi/2-geocentric; x=np.cos(theta)
    instantaneous=BSpline(np.arange(-2300.,2400.,100.),coefficients[0],3)(year)
    north=np.zeros(len(year)); east=north.copy(); down=north.copy(); column=0
    for degree in range(1,4):
        scale=(6371.2/radius)**(degree+2)
        for order in range(degree+1):
            norm=(-1)**order*math.sqrt((1 if order==0 else 2)*math.factorial(degree-order)/math.factorial(degree+order))
            leg=lpmv(order,degree,x)
            previous=lpmv(order,degree-1,x) if order<degree else np.zeros_like(x)
            derivative=-(degree*x*leg-(degree+order)*previous)*np.sin(theta)/(x*x-1)*norm
            leg*=norm
            g=instantaneous[:,column]; column+=1
            h=instantaneous[:,column] if order else 0.
            if order: column+=1
            angular=g*np.cos(order*longitude)+h*np.sin(order*longitude)
            north+=scale*angular*derivative
            east+=scale*order/np.sin(theta)*leg*(g*np.sin(order*longitude)-h*np.cos(order*longitude))
            down-=scale*(degree+1)*angular*leg
    angle=latitude-geocentric
    local_north=north*np.cos(angle)+down*np.sin(angle)
    local_down=down*np.cos(angle)-north*np.sin(angle)
    return local_north,east,local_down

def observation_metrics(coefficients,path=PUBLIC_DATA):
    measured={}
    with np.load(path,allow_pickle=False) as archive:
        for key in ('intensity','inclination','declination'):
            year,latitude,longitude,value,sigma=archive[key][0].T
            local_north,east,local_down=field_at(coefficients,year,latitude,longitude)
            if key=='intensity':prediction=np.sqrt(local_north**2+east**2+local_down**2)
            elif key=='inclination':prediction=np.rad2deg(np.arctan2(local_down,np.hypot(local_north,east)))
            else:prediction=np.rad2deg(np.arctan2(east,local_north))
            residual=value-prediction
            if key=='declination':residual=(residual+180)%360-180
            measured[key+'_whitened_rms']=float(np.sqrt(np.mean((residual/sigma)**2)))
    return measured


def prior_energy(coefficients):
    nodes,weights=np.polynomial.legendre.leggauss(4)
    time=(np.arange(-2000,2000,100)[:,None]+50*(nodes+1)).ravel()
    quadrature=np.tile(weights*50,40)
    spline=BSpline(np.arange(-2300.,2400.,100.),coefficients[0],3)
    q=6371.2/3485.;sw=[];tw=[]
    for n in range(1,4):
        sw.extend([0. if n==1 else 4*np.pi*q**(2*n+3)*(n+1)*(2*n+1)*(2*n+3)/n]*(2*n+1))
        tw.extend([4*np.pi*q**(2*n+4)*(n+1)**2/(2*n+1)]*(2*n+1))
    return float(1e-13*np.sum(quadrature[:,None]*spline(time)**2*np.array(sw))+1e-3*np.sum(quadrature[:,None]*spline(time,nu=2)**2*np.array(tw)))


def energy(coefficients):
    obs=observation_metrics(coefficients)
    with np.load(PUBLIC_DATA,allow_pickle=False) as z:counts={k:z[k].shape[1] for k in ('intensity','inclination','declination')}
    return sum(counts[k]*obs[k+'_whitened_rms']**2 for k in counts)+prior_energy(coefficients)


def map_consistency(fields,coefficients):
    year,lat,lon=np.meshgrid([-1500.,-500.,500.,1000.,1700.],np.linspace(-87.5,87.5,36),np.arange(-177.5,180.,5.),indexing='ij')
    xyz=field_at(coefficients,year.ravel(),lat.ravel(),lon.ravel())
    calculated=np.stack([v.reshape(5,36,72) for v in xyz],axis=1)[None]
    return float(np.max(np.abs(fields-calculated)))


def metrics(estimate,reference):
    a=np.asarray(estimate).ravel(); b=np.asarray(reference).ravel()
    den=np.linalg.norm(a)*np.linalg.norm(b); span=np.ptp(b)
    if np.linalg.norm(b)==0 or span==0: raise ValueError('Degenerate evaluator reference')
    ncc=float(np.dot(a,b)/den) if den>0 else 0.
    result={'ncc':ncc,'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}
    if not all(math.isfinite(value) for value in result.values()): raise ValueError('Computed metric is non-finite')
    return result

def load(path):
    if not Path(path).is_file(): raise ValueError(f'Missing NPZ file: {path}')
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SCHEMA.items():
            if key not in z: raise ValueError(f'Missing required key: {key}')
            a=z[key]
            if a.shape!=shape: raise ValueError(f'{key}: shape {a.shape}; expected {shape}')
            if a.dtype!=np.dtype('float64'): raise ValueError(f'{key}: dtype must be float64')
            if not np.all(np.isfinite(a)): raise ValueError(f'{key}: non-finite values')
            out[key]=a
        return out

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        reference=load(HERE/'baseline_reference.npz');estimate=load(prediction)
        observed=observation_metrics(estimate['coefficients'])
        reference_energy=energy(reference['coefficients'])
        measured=dict(observed,relative_energy_excess=(energy(estimate['coefficients'])-reference_energy)/max(reference_energy,1.),map_consistency_nT=map_consistency(estimate['reconstruction'],estimate['coefficients']))
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Computed metric is non-finite')
        bounds={key:config[key+'_boundary'] for key in measured}
        if not all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in bounds.values()):raise ValueError('Invalid evaluator boundaries')
        report['metrics']=measured;report['thresholds']={key:dict(comparison='<=',boundary=v) for key,v in bounds.items()}
        passed=all(measured[key]<=v for key,v in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=passed)
        return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={}; report['errors']=[str(exc)]; return report,2

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--prediction',required=True); ap.add_argument('--report',required=True)
    args=ap.parse_args(argv); report,code=evaluate(args.prediction)
    try:
        path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr); return 2
    return code
if __name__=='__main__': sys.exit(main())
