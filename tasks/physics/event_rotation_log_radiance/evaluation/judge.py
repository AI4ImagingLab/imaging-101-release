"""Independent standalone evaluator; no solver imports or executable submissions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy import sparse
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,180,240)
def load_array(path,key,shape):
    with np.load(path,allow_pickle=False) as z:
        if key not in z:raise ValueError(f'Missing required key: {key}')
        x=z[key]
    if x.shape!=shape:raise ValueError(f'Expected {key} shape {shape}; received {x.shape}')
    if x.dtype!=np.float64:raise ValueError(f'Expected real float64 {key}; received {x.dtype}')
    if not np.isfinite(x).all():raise ValueError(f'Nonfinite {key}')
    return x
def score(x,r):
    x=x.ravel();r=r.ravel();spread=np.ptp(r);norm=np.linalg.norm(x)*np.linalg.norm(r)
    if spread<=0 or np.linalg.norm(r)<=0:raise ValueError('Degenerate hidden reference')
    return dict(ncc=float(np.dot(x,r)/norm) if norm else 0.,nrmse=float(np.sqrt(np.mean((x-r)**2))/spread))
def observation_residual(image,return_operator=False):
    """Independent forward NIWE check, fitting only unknown positive sensor contrast."""
    path=ASSETS.parent/'data/raw_data.npz'
    ev=load_array(path,'events',(1,27643,4))[0];omega=load_array(path,'angular_velocity',(1,3))[0];K=load_array(path,'intrinsics',(1,3,3))[0]
    fx,fy,cx,cy=np.array([K[0,0],K[1,1],K[0,2],K[1,2]],dtype=np.float32)
    yy,xx=np.mgrid[:180,:240].astype(np.float32);u=xx-cx;v=yy-cy;dt=ev[-1,0]-ev[0,0]
    vx=np.inner(np.stack((u*v/fy,-u*u/fx-fx,fx*v/fy),axis=-1),omega)*dt
    vy=np.inner(np.stack((v*v/fy+fy,-u*v/fx,-fy*u/fx),axis=-1),omega)*dt
    magnitude=np.sqrt(vx*vx+vy*vy);angle=np.arctan2(vy,vx);ux=np.cos(angle);uy=np.sin(angle)
    ux[[0,-1],:]=0;ux[:,[0,-1]]=0;uy[[0,-1],:]=0;uy[:,[0,-1]]=0
    index=ev[:,2].astype(np.int64)*240+ev[:,1].astype(np.int64);fraction=(ev[-1,0]-ev[:,0])/dt
    px=ev[:,1]+fraction*vx.ravel()[index];py=ev[:,2]+fraction*vy.ravel()[index]
    accumulation=np.zeros(43200)
    for oy,ox in [(0,0),(0,1),(1,0),(1,1)]:
        iy=np.floor(py)+oy;ix=np.floor(px)+ox;good=(iy>=0)&(iy<180)&(ix>=0)&(ix<240)
        weight=np.maximum(0,1-np.abs(py-iy))*np.maximum(0,1-np.abs(px-ix))*(2*ev[:,3]-1)
        np.add.at(accumulation,(iy[good]*240+ix[good]).astype(np.int64),weight[good])
    b=accumulation.reshape(180,240)/np.where(magnitude>0,magnitude,1)
    b[[0,-1],:]=0;b[:,[0,-1]]=0
    targetx=xx.astype(float)+ux;targety=yy.astype(float)+uy
    warped=np.zeros_like(image);operator_rows=[];operator_columns=[];operator_values=[]
    for oy,ox in [(0,0),(0,1),(1,0),(1,1)]:
        iy=np.floor(targety).astype(int)+oy;ix=np.floor(targetx).astype(int)+ox
        wgt=np.maximum(0,1-np.abs(targety-iy))*np.maximum(0,1-np.abs(targetx-ix))
        warped+=wgt*image[np.clip(iy,0,179),np.clip(ix,0,239)]
        if return_operator:
            operator_rows.append(np.arange(43200));operator_columns.append((np.clip(iy,0,179)*240+np.clip(ix,0,239)).ravel());operator_values.append(wgt.ravel())
    if return_operator:
        sampler=sparse.coo_matrix((np.concatenate(operator_values),(np.concatenate(operator_rows),np.concatenate(operator_columns))),shape=(43200,43200)).tocsr()
        return sparse.eye(43200,format='csr')-sampler,b.ravel()
    directional=image-warped;norm=np.linalg.norm(directional)
    if norm==0:return 1.
    coefficient=max(0.,float(np.sum(directional*b)/np.sum(b*b)))
    return float(np.linalg.norm(directional-coefficient*b)/norm)

def variation(image):
    return float(np.sum(abs(image[2:]-image[:-2]))/2+np.sum(abs(image[:,2:]-image[:,:-2]))/2)
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'))
        limits={k:config[k+'_boundary'] for k in ['anisotropic_tv','observation_residual']}
        for k,v in limits.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError(f'Invalid {k} boundary')
            if (k=='ncc' and not -1<=v<=1) or (k!='ncc' and v<0):raise ValueError(f'Invalid {k} boundary range')
        x=load_array(Path(prediction),'reconstruction',SHAPE)
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            measured={'anisotropic_tv':variation(x[0]),'observation_residual':observation_residual(x[0])}
        if np.any(x<0) or np.any(x>1) or np.count_nonzero(x==0)<431 or np.count_nonzero(x==1)<431:raise ValueError('Require unit-range percentile gauge with at least431 zero and431 unit samples')
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite metrics')
        thresholds={k:dict(comparison='>=' if k=='ncc' else '<=',boundary=v) for k,v in limits.items()}
        passed=all(measured[k]<=limits[k] for k in limits)
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
