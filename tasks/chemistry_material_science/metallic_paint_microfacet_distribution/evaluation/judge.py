"""Independent evaluator: reads arrays, never imports solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,90),'roughness':(1,2)}
def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        result={}
        for key,shape in SHAPES.items():
            if key not in f: raise ValueError('Missing key '+key)
            a=f[key]
            if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(a).all():
                raise ValueError('Invalid shape, float32/64 dtype, or finite values for '+key)
            result[key]=a.astype(np.float64)
    if np.any(result['reconstruction']<0) or np.any(result['roughness']<=0): raise ValueError('Negative density or nonpositive roughness')
    if result['reconstruction'][0,-1]!=0: raise ValueError('Horizon density sample must equal zero')
    return result
def evaluate(prediction):
    cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
    names=['ncc','nrmse','roughness_error','mass_error','eigen_residual','moment_error']
    thresholds={k:{'comparison':'>=' if k=='ncc' else '<=','boundary':cfg[k+'_boundary']} for k in names}
    if any(not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()): raise ValueError('Invalid metric boundaries')
    refpath=HERE/'ground_truth.npz'
    ref=arrays(refpath if refpath.exists() else HERE/'baseline_reference.npz'); est=arrays(prediction)
    p=est['reconstruction'][0]; r=ref['reconstruction'][0]
    if np.ptp(r)<=0 or np.linalg.norm(r)<=0: raise ValueError('Degenerate reference')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        if np.linalg.norm(p)<=0: raise ValueError('Zero density')
        metrics={'ncc':float(p@r/(np.linalg.norm(p)*np.linalg.norm(r))),
                 'nrmse':float(np.sqrt(np.mean((p-r)**2))/np.ptp(r)),
                 'roughness_error':float(np.max(abs(est['roughness']-ref['roughness'])))}
        t=np.arange(128)/128*np.sqrt(np.pi/2); a=t*t; slope=np.tan(a)
        density=np.interp(np.sqrt(2*np.arctan(abs(slope))/np.pi),np.arange(90)/89,p)
        w=t*density*slope/np.cos(a)**2*(2*np.sqrt(np.pi/2)/128)
        metrics['mass_error']=float(abs(2*np.pi*w.sum()-1))
        moments=np.array([np.sqrt(2*np.pi*np.sum(slope*slope*w)),4*np.sum(slope*w)])
        metrics['moment_error']=float(np.max(abs(moments-est['roughness'][0])))
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f: data=f['brdf'][0]
        angle=(np.arange(89)/89*np.sqrt(np.pi/2))**2
        h=np.where(np.cos(angle)>.99999,0,np.arccos(np.cos(angle)))
        ix=np.minimum(89,np.sqrt(h/(np.pi/2)*8100).astype(int))
        rgb=data[:,ix,0,0].T*np.array([1,1.15,1.66])/1500
        rgb[np.any(rgb<0,axis=1)]=0
        intensity=rgb@np.array([.2126,.7152,.0722])
        s=np.tan(angle); c=np.cos(angle); integral=np.zeros((89,89)); phi=0.
        while phi<2*np.pi:
            integral+=np.maximum(1,s[:,None]*s[None,:]*np.cos(phi)); phi+=np.pi/180
        kernel=(np.sqrt(np.pi/2)/89*c**6*8*intensity)[:,None]*integral*(np.pi/180)*(np.sqrt(angle)*s/c**2)[None,:]
        v=p[:89]; kv=kernel@v; lam=float(v@kv/(v@v))
        metrics['eigen_residual']=float(np.linalg.norm(kv-lam*v)/np.linalg.norm(kv))
    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite metric')
    passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':metrics,'thresholds':thresholds,'errors':[]}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prediction',required=True); parser.add_argument('--report',required=True); args=parser.parse_args()
    try: report=evaluate(Path(args.prediction)); code=0 if report['passed'] else 1
    except Exception as exc: report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc) or type(exc).__name__]}; code=2
    try:
        path=Path(args.report); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc: print('Report write failed: '+str(exc),file=sys.stderr); return 2
    return code
if __name__=='__main__': sys.exit(main())
