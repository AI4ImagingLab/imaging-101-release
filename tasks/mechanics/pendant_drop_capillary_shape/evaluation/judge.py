"""Independent shape, capillary and measured-contour evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import odeint
HERE=Path(__file__).resolve().parent
SHAPES={'parameters':(1,5),'reconstruction':(1,401,2),'capillary_length':(1,1)}
def load(path):
    with np.load(path,allow_pickle=False) as z:
        d={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError('Missing '+k)
            a=z[k]
            if a.shape!=shape:raise ValueError('Invalid shape '+k)
            if a.dtype.kind!='f' or a.dtype.itemsize not in (4,8):raise ValueError('Expected float32 or float64 '+k)
            if not np.isfinite(a).all():raise ValueError('Non-finite '+k)
            d[k]=a.astype(float)
    p=d['parameters'][0]
    if p[2]<=0 or p[3]<=0 or d['capillary_length'][0,0]<=0:raise ValueError('Radius, Bond and capillary length must be positive')
    return d
def distance_rms(points,curve):
    start=curve[:-1];v=np.diff(curve,axis=0);v2=np.sum(v*v,axis=1)
    if np.any(v2<=0):raise ValueError('Degenerate silhouette segments')
    sums=[]
    for chunk in np.array_split(points,16):
        delta=chunk[:,None,:]-start[None]
        t=np.clip(np.sum(delta*v[None],axis=2)/v2,0,1)
        residual=delta-t[:,:,None]*v[None]
        sums.extend(np.min(np.sum(residual*residual,axis=2),axis=1))
    return float(np.sqrt(np.mean(sums)))
def physical_curve(p):
    x,y,R,B,angle=p
    def rhs(v,s):
        r,z,phi=v
        return [np.cos(phi),np.sin(phi),2-B*z-np.sin(phi)/r]
    a=odeint(rhs,[1e-6,0,0],np.linspace(0,4,4001))
    rz=np.concatenate((a[:0:-1,:2]*[-1,1],a[:,:2]))*R
    r,z=rz.T;c=np.cos(angle);s=np.sin(angle)
    return np.stack((x+c*r-s*z,y-s*r-c*z),axis=1)
def evaluate(prediction):
    out=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        truth=HERE/'ground_truth.npz';ref=load(truth if truth.exists() else HERE/'baseline_reference.npz')
        limits=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        names=('geometry_max','bond_error','angle_error','profile_rmse','capillary_error','capillary_consistency','forward_rms','observation_rms')
        for n in names:
            b=limits[n+'_boundary']
            if not isinstance(b,(int,float)) or not np.isfinite(b) or b<0:raise ValueError('Invalid boundary '+n)
            out['thresholds'][n]={'comparison':'<=','boundary':b}
        d=load(prediction);p=d['parameters'][0];r=ref['parameters'][0];curve=d['reconstruction'][0]
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:obs=z['contour'][0]
        if obs.shape!=(293,2) or not np.isfinite(obs).all():raise ValueError('Invalid observation asset')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            m=dict(geometry_max=float(np.max(abs(p[:3]-r[:3]))),bond_error=float(abs(p[3]-r[3])),angle_error=float(abs(p[4]-r[4])),
                   profile_rmse=float(np.sqrt(np.mean((curve-ref['reconstruction'][0])**2))),
                   capillary_error=float(abs(d['capillary_length'][0,0]-ref['capillary_length'][0,0])),
                   capillary_consistency=float(abs(d['capillary_length'][0,0]-p[2]/np.sqrt(p[3]))))
            # A failed numerical contract already proves failure; avoid pathological
            # ODEs and report only actually measured metrics, never sentinel values.
            if not all(np.isfinite(v) for v in m.values()):raise ValueError('Non-finite metric')
            if any(m[n]>limits[n+'_boundary'] for n in m):
                out.update(status='fail',metrics=m);return out
            m.update(forward_rms=distance_rms(curve,physical_curve(p)),observation_rms=distance_rms(obs,curve))
        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Non-finite metric')
        out['metrics']=m;out['passed']=all(m[n]<=limits[n+'_boundary'] for n in names);out['status']='pass' if out['passed'] else 'fail'
    except Exception as e:out['metrics']={};out['errors']=[str(e)];out['status']='error';out['passed']=False
    return out
def main(argv=None):
    a=argparse.ArgumentParser();a.add_argument('--prediction',required=True);a.add_argument('--report',required=True);args=a.parse_args(argv)
    r=evaluate(Path(args.prediction))
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return {'pass':0,'fail':1,'error':2}[r['status']]
if __name__=='__main__':raise SystemExit(main())
