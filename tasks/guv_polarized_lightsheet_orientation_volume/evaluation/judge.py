"""Standalone coefficient-field Judge; no solver, submitted-code or domain imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import sph_harm_y
HERE=Path(__file__).resolve().parent
SHAPE=(1,77,55,80,15)

def load_field(path):
    if not Path(path).is_file():raise ValueError('NPZ file does not exist')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Require exactly reconstruction')
        x=z['reconstruction']
    if x.shape!=SHAPE:raise ValueError('Wrong batch-first coefficient shape')
    if x.dtype!=np.float64:raise ValueError('Require real float64 coefficients')
    if not np.isfinite(x).all():raise ValueError('Nonfinite coefficients')
    return x[0]

def known_operator_and_observations():
    """Independently evaluate the known dipole/finite-NA model from public inputs.

    Angular multiplication uses direct surface integration. Calibration angles
    and volume factors use the acquisition lake only, without target fitting.
    """
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        counts=z['observations'];lake=z['lake_calibration']
    if counts.shape!=(1,77,55,80,21,2) or lake.shape!=(1,30,30,30,21,2):raise ValueError('Invalid public acquisition shape')
    if counts.dtype!=np.uint16 or lake.dtype!=np.uint16:raise ValueError('Invalid public acquisition dtype')
    lake=lake[0].astype(np.float32);means=lake.mean(axis=(0,1,2))
    a=np.deg2rad([0,45,60,90,120,135,180]);design=np.column_stack([np.ones(7),np.cos(2*a),np.sin(2*a)])
    grid=np.linspace(0,180,1000);offsets=[]
    for v in range(2):
        for t in range(3):
            y=lake[...,7*t:7*(t+1),v].mean(axis=(0,1,2));b=np.linalg.lstsq(design,y,rcond=None)[0]
            offsets.append(grid[np.argmax(b[0]+b[1]*np.cos(2*np.deg2rad(grid))+b[2]*np.sin(2*np.deg2rad(grid)))])
    theta=np.deg2rad(np.array([0,45,60,90,120,135,180])-np.mean(offsets));pol=np.empty((2,21,3))
    for k,(ta,tb) in enumerate([(0,0),(8.9,7.7),(-9.2,-6.5)]):
        ta,tb=np.deg2rad([ta,tb]);c,s=np.cos(theta),np.sin(theta)
        pol[0,7*k:7*(k+1)]=np.stack([np.sin(ta)*c,np.cos(ta)*c,-s],axis=-1)
        pol[1,7*k:7*(k+1)]=np.stack([-s,np.cos(tb)*c,np.sin(tb)*c],axis=-1)
    take=np.delete(np.arange(21),1);pol=pol[:,take];pol/=np.linalg.norm(pol,axis=-1,keepdims=True)
    nodes,weights=np.polynomial.legendre.leggauss(12);az=np.arange(32)*2*np.pi/32
    polar=np.arccos(nodes)[:,None];az=az[None,:]
    Y=[]
    for ell in (0,2,4):
        for order in range(-ell,ell+1):
            y=sph_harm_y(ell,order,polar,az)
            if order>0:y=(y+(-1.)**order*sph_harm_y(ell,-order,polar,az))/np.sqrt(2)
            elif order<0:y=(y-(-1.)**order*sph_harm_y(ell,-order,polar,az))/(1j*np.sqrt(2))
            Y.append(y.real.ravel())
    Y=np.asarray(Y);w=np.repeat(weights,32)*2*np.pi/32
    C=np.einsum('in,jn,kn,n->ijk',Y,Y[:6],Y[[0,3,5]],w);C[abs(C)<1e-14]=0
    def sh(l,m,t,p):
        y=sph_harm_y(l,m,t,p)
        if m>0:y=(y+(-1.)**m*sph_harm_y(l,-m,t,p))/np.sqrt(2)
        elif m<0:y=(y-(-1.)**m*sph_harm_y(l,-m,t,p))/(1j*np.sqrt(2))
        return y.real
    qtheta=np.arccos(pol[...,2]);qphi=np.arctan2(pol[...,1],pol[...,0])
    illumination=np.stack([sh(l,m,qtheta,qphi)*(1 if l==0 else .4) for l in (0,2) for m in range(-l,l+1)],axis=-1)
    R=np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,-.5,0,np.sqrt(3)/2],[0,0,0,0,1,0],[0,0,0,np.sqrt(3)/2,0,.5]])
    h={}
    for view,na in enumerate((1.1,.67)):
        axes=[np.fft.rfftfreq(n,130)*561/na for n in (77,55,80)]
        aa,bb=(axes[0],axes[1]) if view==0 else (axes[1],axes[2])
        r=np.hypot(aa[:,None],bb[None,:]);u=np.minimum(r/2,1);root=u*np.sqrt(1-u*u)
        a1=2/np.pi*(np.arccos(u)-root);a2=2/np.pi*(np.arccos(u)-(3-r*r/2)*root)
        det=np.zeros(r.shape+(6,));det[...,0]=a1+(na/1.33)**2/4*a2;det[...,3]=(-a1+(na/1.33)**2/2*a2)/np.sqrt(5)
        det0=np.array([1+(na/1.33)**2/4,0,0,(-1+(na/1.33)**2/2)/np.sqrt(5),0,0])
        ill0=np.array([1,0,0,-1/np.sqrt(5),0,0])/np.sqrt(4*np.pi)
        if view:det=det@R.T;det0=det0@R.T
        else:ill0=ill0@R.T
        norm=np.einsum('jk,j,k->',C[0],ill0,det0[[0,3,5]])
        H=np.einsum('ijk,pj,xyk->xyip',C,illumination[view],det[...,[0,3,5]])/norm
        H=H.astype(np.float32);H/=np.max(abs(H))
        h['xy' if view==0 else 'yz']=H
        freqs=axes[2] if view==0 else axes[0]
        h['z' if view==0 else 'x']=np.exp(-freqs*freqs/(2*.25**2),dtype=np.float32) if view==0 else np.exp(-freqs*freqs/(2*.25**2))
    # The uniform f00 unit specimen has only the zero-frequency bin.
    V=77*55*80
    model=np.stack([np.float32(h['xy'][0,0,0]*V),np.float32(h['yz'][0,0,0]*V)],axis=-1).astype(float)/V
    model/=np.max(model)
    model=np.broadcast_to(model,(77,55,80,20,2)).mean(axis=(0,1,2))
    g=(counts[0]/65535).astype(np.float32);g=g[...,take,:]
    g=g*means[take[0],0]*model/means[take,:]
    return h,g

def observation_residual(field,h,g):
    F=np.fft.rfftn(field,axes=(0,1,2));G=np.empty((77,55,41,20,2),np.complex64)
    ix=np.abs(np.rint(np.fft.fftfreq(77)*77)).astype(int);iy=np.abs(np.rint(np.fft.fftfreq(55)*55)).astype(int)
    for z in range(41):
        H0=h['z'][z]*h['xy'][ix[:,None],iy[None,:]]
        H1=h['x'][ix,None,None,None]*h['yz'][iy[None,:],z]
        G[:,:,z,:,0]=np.einsum('xyjp,xyj->xyp',H0,F[:,:,z])
        G[:,:,z,:,1]=np.einsum('xyjp,xyj->xyp',H1,F[:,:,z])
    pred=np.fft.irfftn(G.astype(complex),s=(77,55,80),axes=(0,1,2))
    return float(np.linalg.norm(pred-g)/np.linalg.norm(g))

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        keys=['ncc_boundary','nrmse_boundary','max_abs_boundary','forward_l2_boundary']
        for k in keys:
            v=cfg[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Undefined metric boundary '+k)
        if not -1<=cfg['ncc_boundary']<=1 or any(cfg[k]<0 for k in keys[1:]):raise ValueError('Invalid metric boundary range')
        truth=HERE/'ground_truth.npz';reference=load_field(truth if truth.exists() else HERE/'baseline_reference.npz')
        if any(np.ptp(reference[...,j])==0 or np.linalg.norm(reference[...,j])==0 for j in range(15)):raise ValueError('Degenerate reference coefficient')
        h,g=known_operator_and_observations()
    except Exception as e:report['errors']=['Evaluator error: '+str(e)];return report
    for j in range(15):
        for name,bound,direction in [('ncc',cfg['ncc_boundary'],'>='),('nrmse',cfg['nrmse_boundary'],'<='),('max_abs',cfg['max_abs_boundary'],'<=')]:
            report['thresholds'][f'coefficient_{j}_{name}']={'boundary':bound,'comparison':direction}
    report['thresholds']['forward_l2']={'boundary':cfg['forward_l2_boundary'],'comparison':'<='}
    try:
        field=load_field(prediction);metrics={}
        for j in range(15):
            x=field[...,j].ravel();r=reference[...,j].ravel();norm=np.linalg.norm(x);e=x-r
            metrics[f'coefficient_{j}_ncc']=float(x@r/norm/np.linalg.norm(r)) if norm else 0.
            metrics[f'coefficient_{j}_nrmse']=float(np.sqrt(np.mean(e*e))/np.ptp(r))
            metrics[f'coefficient_{j}_max_abs']=float(np.max(abs(e)))
        metrics['forward_l2']=observation_residual(field,h,g)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Metric overflow')
    except Exception as e:report['errors']=['Invalid prediction: '+str(e)];return report
    passed=all(v>=report['thresholds'][k]['boundary'] if report['thresholds'][k]['comparison']=='>=' else v<=report['thresholds'][k]['boundary'] for k,v in metrics.items())
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    return report

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    report=evaluate(a.prediction)
    try:
        dest=Path(a.report);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
