"""Standalone canonical SE2 evaluation; imports no solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent

def score(x,reference,raw):
    pos=float(np.max(np.linalg.norm(x[0,:,:2]-reference[0,:,:2],axis=1)))
    ang=float(np.max(np.abs((x[0,:,2]-reference[0,:,2]+np.pi)%(2*np.pi)-np.pi)))
    ids=raw['edge_indices'][0];z=raw['relative_poses'][0];o=raw['information'][0]
    a=x[0,ids[:,0]];b=x[0,ids[:,1]];delta=b[:,:2]-a[:,:2];c=np.cos(a[:,2]);s=np.sin(a[:,2])
    q=np.column_stack((c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]))-z[:,:2]
    cz=np.cos(z[:,2]);sz=np.sin(z[:,2]);e=np.column_stack((cz*q[:,0]+sz*q[:,1],-sz*q[:,0]+cz*q[:,1],(b[:,2]-a[:,2]-z[:,2]+np.pi)%(2*np.pi)-np.pi))
    return {'position_max_error_m':pos,'heading_max_error_rad':ang,'weighted_observation_error':float(np.einsum('ni,nij,nj->',e,o,e))}

def evaluate(prediction,assets=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        cfg=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        for key in ['position_max_error_m','heading_max_error_rad','weighted_observation_error']:
            value=cfg[key+'_boundary']
            if not isinstance(value,(float,int)) or not np.isfinite(value) or value<=0: raise ValueError('Invalid boundary '+key)
            report['thresholds'][key]={'comparison':'<=','boundary':value}
        refpath=assets/'ground_truth.npz'
        if not refpath.exists(): refpath=assets/'baseline_reference.npz'
        with np.load(refpath,allow_pickle=False) as z: ref=z['reconstruction'].copy()
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as z: raw={k:z[k].copy() for k in ['edge_indices','relative_poses','information']}
        if ref.shape!=(1,1228,3) or ref.dtype!=np.float64 or not np.isfinite(ref).all(): raise ValueError('Invalid hidden reference')
        with np.load(prediction,allow_pickle=False) as z:
            if set(z.files)!={'reconstruction'}: raise ValueError('Required exact key: reconstruction')
            x=z['reconstruction'].copy()
        if x.shape!=(1,1228,3): raise ValueError('Expected shape (1,1228,3)')
        if x.dtype!=np.float64: raise ValueError('Expected real float64 dtype')
        if not np.isfinite(x).all(): raise ValueError('Nonfinite reconstruction')
        if np.any(x[0,:,2]<=-np.pi) or np.any(x[0,:,2]>np.pi): raise ValueError('Heading must be in (-pi,pi]')
        if not np.array_equal(x[0,0],np.zeros(3)): raise ValueError('Anchor must equal identity exactly')
        metrics=score(x,ref,raw)
        if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Undefined metric')
        report['metrics']=metrics;report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in metrics.items());report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as error:
        report['errors']=[str(error)];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    result,code=evaluate(Path(args.prediction))
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error: print('Report write failed: '+str(error),file=sys.stderr);return 2
    return code

if __name__=='__main__': sys.exit(main())
