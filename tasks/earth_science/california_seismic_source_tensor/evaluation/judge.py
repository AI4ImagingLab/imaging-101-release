"""Independent standalone evaluator; never imports the submitted/reference solver."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def load_checked(path,shapes):
    if not Path(path).is_file(): raise ValueError(f'File unavailable: {path}')
    with np.load(path,allow_pickle=False) as z:
        result={}
        for key,shape in shapes.items():
            if key not in z: raise ValueError(f'Missing key: {key}')
            a=z[key]
            if a.shape!=shape: raise ValueError(f'{key}: expected shape{shape}, received{a.shape}')
            if a.dtype!=np.dtype('float64'): raise ValueError(f'{key}: dtype must be float64')
            if not np.all(np.isfinite(a)): raise ValueError(f'{key}: nonfinite value')
            result[key]=a
    return result

def observation_vr(tensor,depth):
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        gf=z['fundamental_greens'][0,[10.,12.,20.].index(float(depth))].astype(np.float64)
        obs=z['waveforms'][0]
        az=z['station_azimuth'][0]*np.pi/180
        weight=z['station_distance'][0]/np.min(z['station_distance'][0])
    shifts=[32,32,31,31,31,31,31,32]
    d=np.stack([obs[i,:,s:s+150] for i,s in enumerate(shifts)]).astype(np.float64)
    M=tensor/1e13;pred=np.zeros(d.shape)
    for i,alpha in enumerate(az):
        css=.5*(M[0,0]-M[1,1])*np.cos(2*alpha)+M[0,1]*np.sin(2*alpha)
        cds=M[0,2]*np.cos(alpha)+M[1,2]*np.sin(alpha)
        cdd=-.166667*(M[0,0]+M[1,1])+.33333*M[2,2]
        cex=.33333*np.trace(M)
        for c,off in [(0,0),(1,4)]:
            pred[i,c]=css*gf[i,off]+cds*gf[i,off+1]+cdd*gf[i,off+2]+cex*gf[i,off+3]
        pred[i,2]=(.5*(M[0,0]-M[1,1])*np.sin(2*alpha)-M[0,1]*np.cos(2*alpha))*gf[i,8]+(M[0,2]*np.sin(alpha)-M[1,2]*np.cos(alpha))*gf[i,9]
    w=weight[:,None,None]
    return float(100*(1-np.sum(w*(d-pred)**2)/np.sum(w*d*d)))

def evaluate(prediction):
    shapes={'reconstruction':(1,3,3),'source_depth':(1,1)}
    limits=json.loads((HERE/'metrics.json').read_text())
    thresholds={k:{'comparison':direction,'boundary':limits[k+'_boundary']} for k,direction in [('tensor_relative_error','<='),('depth_error_km','<='),('variance_reduction','>=')]}
    ref=load_checked(HERE/'baseline_reference.npz',shapes)
    z=load_checked(prediction,shapes)
    tensor=z['reconstruction'][0];depth=float(z['source_depth'][0,0])
    if not np.array_equal(tensor,tensor.T): raise ValueError('Moment tensor must be symmetric')
    if depth not in (10.,12.,20.): raise ValueError('Source depth must be one of10,12,20km')
    norm=np.linalg.norm(ref['reconstruction'])
    if not np.isfinite(norm) or norm<=0: raise ValueError('Invalid hidden reference tensor norm')
    metrics={'tensor_relative_error':float(np.linalg.norm(z['reconstruction']-ref['reconstruction'])/norm),'depth_error_km':float(abs(depth-ref['source_depth'][0,0])),'variance_reduction':observation_vr(tensor,depth)}
    if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Undefined evaluator metric')
    passed=all(metrics[k]<=v['boundary'] if v['comparison']=='<=' else metrics[k]>=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':metrics,'thresholds':thresholds,'errors':[]},0 if passed else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    try: report,code=evaluate(args.prediction)
    except Exception as exc: report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__': sys.exit(main())
