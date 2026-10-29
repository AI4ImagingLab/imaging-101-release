"""Standalone deterministic NPZ evaluator; no solver imports or execution."""
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,128,128,96),'r1':(1,128,128,96)}

def load_arrays(path):
    if not Path(path).is_file(): raise ValueError(f'File does not exist: {path}')
    with np.load(path,allow_pickle=False) as archive:
        result={}
        for key,shape in SHAPES.items():
            if key not in archive: raise ValueError(f'Missing required key: {key}')
            value=archive[key]
            if value.shape!=shape: raise ValueError(f'{key}: expected shape {shape}, received {value.shape}')
            if value.dtype!=np.dtype('float64'): raise ValueError(f'{key}: expected real float64, received {value.dtype}')
            if not np.isfinite(value).all(): raise ValueError(f'{key}: non-finite values')
            result[key]=value
    return result

def metrics(prediction,reference):
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:
        p,t,m=(f[k][0] for k in ('PDw','T1w','MTw'))
    x,y,z=np.indices((128,128,96));u=p/(6*np.pi/180);v=t/(20*np.pi/180)
    mask=(((x-63)/38)**2+((y-64)/53)**2+((z-48)/30)**2<1)&(p>100)&(t>100)&(m>50)&(np.abs(u-v)/(u+v+1e-100)>.2)
    measured={}
    for k in ('reconstruction','r1'):
        a=prediction[k][0][mask];b=reference[k][0][mask]
        if np.linalg.norm(b)==0: raise ValueError('Degenerate reference')
        measured[k+'_rmse']=float(np.sqrt(np.mean((a-b)**2)))
        measured[k+'_ncc']=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.
    measured['outside_max']=max(float(np.max(np.abs(prediction[k][0][~mask]))) for k in prediction)
    r=prediction['r1'][0][mask];delta=prediction['reconstruction'][0][mask]/100
    alpha=np.array([6.,20.,6.])*np.pi/180;tr=np.array([.028,.018,.028])
    if np.any(r==0): measured['observation_relative_rmse']=1e100
    else:
        amp=p[mask]*(tr[0]*r+alpha[0]**2/2)/(alpha[0]*tr[0]*r)
        den=np.stack([tr[i]*r+alpha[i]**2/2+(delta if i==2 else 0) for i in range(3)])
        if np.any(den==0): measured['observation_relative_rmse']=1e100
        else:
            predicted=amp[None]*alpha[:,None]*tr[:,None]*r[None]/den
            obs=np.stack([p[mask],t[mask],m[mask]])
            measured['observation_relative_rmse']=float(np.linalg.norm(predicted-obs)/np.linalg.norm(obs))
    return measured

def evaluate(prediction_path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        thresholds=cfg['thresholds']
        expected={'reconstruction_rmse','r1_rmse','reconstruction_ncc','r1_ncc','outside_max','observation_relative_rmse'}
        if set(thresholds)!=expected: raise ValueError('Undefined or missing metric definitions')
        if any(v['comparison']!=('>=' if k.endswith('_ncc') else '<=') for k,v in thresholds.items()): raise ValueError('Invalid metric direction')
        if any(not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):
            raise ValueError('Invalid metric boundaries')
        report['thresholds']=thresholds
        refpath=HERE/'ground_truth.npz'
        if not refpath.exists(): refpath=HERE/'baseline_reference.npz'
        ref=load_arrays(refpath);pred=load_arrays(prediction_path)
        measured=metrics(pred,ref)
        if not all(np.isfinite(v) for v in measured.values()): raise ValueError('Non-finite evaluation metric')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        report['metrics']={}
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Report write failed: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
