"""Standalone independent NumPy orientation/observation evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys,itertools
import numpy as np
ASSETS=Path(__file__).resolve().parent;SHAPE=(1,40,50,4)
def load_prediction(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        q=z['reconstruction']
    if q.shape!=SHAPE or q.dtype!=np.float32:raise ValueError('Require float32 reconstruction(1,40,50,4)')
    if not np.isfinite(q).all():raise ValueError('Nonfinite quaternion')
    q=q.astype(np.float64);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.max(abs(norm-1))>1e-5:raise ValueError('Quaternions must have unit norm within1e-5')
    return q/norm
def multiply(a,b):
    w=a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1)
    v=a[...,0,None]*b[...,1:]+b[...,0,None]*a[...,1:]+np.cross(a[...,1:],b[...,1:])
    return np.concatenate([w[...,None],v],axis=-1)
def cubic_symmetry():
    """Independent24 proper cube rotations from axis-angle geometric generators."""
    group=[[1.,0,0,0]]
    for axis in np.eye(3):
        for angle in [np.pi/2,np.pi,3*np.pi/2]:group.append([np.cos(angle/2),*(axis*np.sin(angle/2))])
    for yz in itertools.product([-1.,1.],repeat=2):
        axis=np.array([1.,*yz])/np.sqrt(3)
        for angle in [2*np.pi/3,4*np.pi/3]:group.append([np.cos(angle/2),*(axis*np.sin(angle/2))])
    for zero in range(3):
        for sign in [-1.,1.]:
            axis=np.zeros(3);ids=[i for i in range(3) if i!=zero];axis[ids]=[1,sign];axis/=np.sqrt(2);group.append([0.,*axis])
    return np.array(group)
def disorientation_degrees(q,r):
    """Physical crystal symmetry acts on the left for sample-ray->crystal q."""
    a=np.asarray(q,dtype=np.float64).reshape(-1,4);b=np.asarray(r,dtype=np.float64).reshape(-1,4);a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    equivalent=multiply(cubic_symmetry()[None],b[:,None]);dot=np.max(abs(np.einsum('ij,ikj->ik',a,equivalent)),axis=-1)
    return 2*np.degrees(np.arccos(np.clip(dot,0,1)))
def rotate(q,rays):
    v=q[...,1:];cross=np.cross(v[...,None,:],rays);return rays+2*(q[...,0,None,None]*cross+np.cross(v[...,None,:],cross))
def lambert(rays):
    x,y,z=np.moveaxis(rays,-1,0);factor=np.sqrt(np.maximum(0,2*(1-abs(z))));cond=abs(y)<=abs(x);big=np.where(cond,x,y);small=np.where(cond,y,x);sign=np.where(big<0,-1.,1.)
    a=sign*factor/np.sqrt(2);b=sign*factor*np.arctan2(small*sign,abs(big))*2*np.sqrt(2)/np.pi
    u=np.where(cond,a,b)/2-.5+(z<0);v=np.where(cond,b,a);return u,v
def bilinear(master,u,v):
    """Independent scalar bilinear sampling with align_corners=True, border pad."""
    x=np.clip((u+1)*(master.shape[1]-1)/2,0,master.shape[1]-1);y=np.clip((v+1)*(master.shape[0]-1)/2,0,master.shape[0]-1);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,master.shape[1]-1);y1=np.minimum(y0+1,master.shape[0]-1);dx=x-x0;dy=y-y0
    return (1-dy)*((1-dx)*master[y0,x0]+dx*master[y0,x1])+dy*((1-dx)*master[y1,x0]+dx*master[y1,x1])
def clahe(x,grid_shape):
    """Independent histogram/CDF/trilinear implementation of published preprocessing."""
    x=np.asarray(x);original_dtype=x.dtype;B,H,W=x.shape;gh,gw=grid_shape;th=int(np.ceil(H/gh));tw=int(np.ceil(W/gw));p=np.pad(x,((0,0),(0,gh*th-H),(0,gw*tw-W)),mode='reflect');tiles=p.reshape(B,gh,th,gw,tw).transpose(0,1,3,2,4).reshape(B,gh*gw,-1);vox=th*tw
    discrete=(tiles*np.array(63,dtype=original_dtype)).astype(original_dtype).astype(int);hist=np.zeros((B,gh*gw,64),dtype=np.float32)
    for batch in range(B):
        for tile in range(gh*gw):hist[batch,tile]=np.bincount(discrete[batch,tile],minlength=64)
    hist=(hist/np.float32(vox))*np.float32(vox);limit=max(40.*vox//64,1);hist=np.minimum(hist,limit);clipped=vox-hist.sum(-1);residual=np.remainder(clipped,64);hist+=(clipped-residual)[...,None]/64;hist+=(np.arange(64)[None,None,:]<residual[...,None]);cdf=np.clip(np.cumsum(hist,axis=-1,dtype=np.float32)/vox,0,1).reshape(B,gh,gw,64)
    # Source performs intensity*2-1 in its current FP16/FP32 dtype first.
    value=((x*np.array(2,dtype=original_dtype)).astype(original_dtype)-np.array(1,dtype=original_dtype)).astype(original_dtype).astype(np.float32);coord=(value+1)*32-.5
    rr=np.linspace(-1,1,H,dtype=np.float32);cc=np.linspace(-1,1,W,dtype=np.float32);row=np.clip((rr+1)*gh/2-.5,0,gh-1);col=np.clip((cc+1)*gw/2-.5,0,gw-1);coord=np.clip(coord,0,63);r0=np.floor(row).astype(int);c0=np.floor(col).astype(int);v0=np.floor(coord).astype(int);r1=np.minimum(r0+1,gh-1);c1=np.minimum(c0+1,gw-1);v1=np.minimum(v0+1,63);dr=row-r0;dc=col-c0;dv=coord-v0;out=np.zeros((B,H,W),dtype=np.float32);batch=np.arange(B)[:,None,None]
    for ir,wr in [(r0,1-dr),(r1,dr)]:
        for ic,wc in [(c0,1-dc),(c1,dc)]:
            for iv,wv in [(v0,1-dv),(v1,dv)]:out+=(cdf[batch,ir[None,:,None],ic[None,None,:],iv]*wr[None,:,None]*wc[None,None,:]*wv).astype(np.float32)
    return out
def minmax(a):
    shape=a.shape;a=a.reshape(len(a),-1);lo=a.min(-1,keepdims=True);hi=a.max(-1,keepdims=True);return ((a-lo)/(np.array(1e-4,dtype=a.dtype)+hi-lo)).reshape(shape)
def observation_assets():
    with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as z:raw=z['patterns'];m=z['master_pattern']
    if raw.shape!=(1,40,50,60,60) or raw.dtype!=np.uint8 or m.shape!=(1,2,401,401) or m.dtype!=np.uint8:raise ValueError('Invalid public observation/calibration')
    p=raw[0].reshape(2000,30,2,30,2).mean((2,4),dtype=np.float32).astype(np.float16);p=(p-p.mean(0,keepdims=True,dtype=np.float32).astype(np.float16)).astype(np.float16);p=minmax(p);p=clahe(p,(4,4));p=minmax(p).reshape(2000,900);p=p-p.mean(-1,keepdims=True)
    master=np.concatenate([m[0,0],m[0,1]],axis=-1).astype(np.float32);master=(master-master.min())/(np.float32(1e-4)+master.max()-master.min());master=clahe(master[None],(8,8))[0]
    # Direct independently derived Bruker ray formula, square detector,70deg tilt.
    phi=np.deg2rad(-70.);i,j=np.meshgrid(np.arange(30),np.arange(30),indexing='ij');a=i+.5-.2179*30;b=j+.5-.4221*30;D=.4954*30;rays=np.stack([-np.sin(phi)*a+np.cos(phi)*D,b,-np.cos(phi)*a-np.sin(phi)*D],-1).reshape(900,3);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    return p,master,rays
def forward_cosines(q,assets=None):
    """Best public-calibration cosine over physical cubic representatives per pixel.

    Maximizing over all24 allows physical symmetry despite intensity preprocessing
    not being exactly equivariant under cube rotations on a finite master grid.
    """
    if assets is None:assets=observation_assets()
    obs,master,rays=assets;q=np.array(q,dtype=float,copy=True).reshape(-1,4);q/=np.linalg.norm(q,axis=-1,keepdims=True);group=cubic_symmetry();scores=[]
    for start in range(0,len(q),16):
        batch=q[start:start+16];proposal=multiply(group[None],batch[:,None]);v=rotate(proposal,rays);u,w=lambert(v);sim=bilinear(master,u,w);sim-=sim.mean(-1,keepdims=True);y=obs[start:start+len(batch)];den=np.linalg.norm(sim,axis=-1)*np.linalg.norm(y,axis=-1)[:,None];cor=np.divide(np.einsum('ij,ikj->ik',y,sim),den,out=np.zeros_like(den),where=den>0);scores.extend(cor.max(-1))
    return np.array(scores)
def contrast_assets():
    """Simple observation-defined contrast; no PCA/CLAHE/reference solver."""
    with np.load(ASSETS.parent/'data/raw_data.npz',allow_pickle=False) as z:
        raw=z['patterns'];m=z['master_pattern']
    if raw.shape!=(1,40,50,60,60) or raw.dtype!=np.uint8 or m.shape!=(1,2,401,401) or m.dtype!=np.uint8:raise ValueError('Invalid observation/calibration arrays')
    p=raw[0].reshape(2000,30,2,30,2).mean((2,4),dtype=float)
    p-=p.mean(axis=0);p=p.reshape(2000,900);p-=p.mean(axis=1,keepdims=True)
    master=np.concatenate([m[0,0],m[0,1]],axis=-1).astype(float)
    phi=np.deg2rad(-70.);i,j=np.meshgrid(np.arange(30),np.arange(30),indexing='ij')
    a=i+.5-.2179*30;b=j+.5-.4221*30;D=.4954*30
    rays=np.stack([-np.sin(phi)*a+np.cos(phi)*D,b,-np.cos(phi)*a-np.sin(phi)*D],-1).reshape(900,3);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    return p,master,rays


def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((ASSETS/'metrics.json').read_text(encoding='utf8'))
        limits={k:cfg[k+'_boundary'] for k in ('mean_forward_cosine','maximum_contrast_regret')}
        attained=np.asarray(cfg['attained_pattern_cosines'],dtype=float)
        if attained.shape!=(2000,) or not np.isfinite(attained).all() or np.any(abs(attained)>1):raise ValueError('Invalid attainable contrast values')
        for key,value in limits.items():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):raise ValueError('Invalid boundary '+key)
            if key=='maximum_contrast_regret' and not 0<=value<=1:raise ValueError('Invalid contrast budget')
            if key=='mean_forward_cosine' and not -1<=value<=1:raise ValueError('Invalid mean cosine boundary')
        q=load_prediction(prediction);cosine=forward_cosines(q,contrast_assets())
        measured={'mean_forward_cosine':float(cosine.mean()),'maximum_contrast_regret':float(np.max(attained-cosine))}
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite metric')
        passed=measured['mean_forward_cosine']>=limits['mean_forward_cosine'] and measured['maximum_contrast_regret']<=limits['maximum_contrast_regret']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds={k:dict(comparison='>=' if k=='mean_forward_cosine' else '<=',boundary=v) for k,v in limits.items()})
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv);report,code=evaluate(a.prediction)
    try:
        target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
