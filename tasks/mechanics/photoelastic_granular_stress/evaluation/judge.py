"""Standalone strict NPZ evaluator; no solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSET_DIR=Path(__file__).resolve().parent

def agreement(estimate,reference):
    e=estimate.ravel();r=reference.ravel()
    en=np.linalg.norm(e);rn=np.linalg.norm(r);span=float(np.ptp(r))
    if rn==0 or span==0: raise ValueError('Undefined reference metric: zero norm or dynamic range')
    ncc=float(np.dot(e,r)/(en*rn)) if en else 0.
    return ncc,float(np.sqrt(np.mean((e-r)**2))/span)

def evaluate(prediction,asset_dir=ASSET_DIR):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'))
        target=asset_dir/'ground_truth.npz'
        if not target.exists(): target=asset_dir/'baseline_reference.npz'
        if not target.is_file(): raise ValueError('Evaluator reference unavailable')
        with np.load(target,allow_pickle=False) as f: refs={k:f[k] for k in config['output_contract']}
        if not Path(prediction).is_file(): raise ValueError('Prediction file does not exist')
        with np.load(prediction,allow_pickle=False) as f:
            estimates={}
            for k,c in config['output_contract'].items():
                if k not in f: raise ValueError('Missing prediction key: '+k)
                v=f[k]
                if list(v.shape)!=c['shape']: raise ValueError('Wrong prediction shape: '+k)
                if v.dtype!=np.dtype(c['dtype']): raise ValueError('Wrong prediction dtype: '+k)
                if not np.isfinite(v).all(): raise ValueError('Nonfinite prediction: '+k)
                r=refs[k]
                if r.shape!=v.shape or r.dtype!=v.dtype or not np.isfinite(r).all(): raise ValueError('Invalid evaluator reference: '+k)
                estimates[k]=v
        metrics={};thresholds={}
        for k,v in estimates.items():
            nc,nr=agreement(v,refs[k]);metrics[k+'_ncc']=nc;metrics[k+'_nrmse']=nr
            thresholds[k+'_ncc']={'comparison':'>=','boundary':config['ncc_boundary']}
            thresholds[k+'_nrmse']={'comparison':'<=','boundary':config['nrmse_boundary']}
        # Independent static force/torque audit in the public contact convention.
        force=estimates['forces'][0];alpha=estimates['contact_directions'][0]
        beta=np.array([0.,np.pi/2,np.pi,-np.pi/2]);total=float(np.sum(force))
        net=np.array([np.sum(force*np.cos(beta-alpha)),-np.sum(force*np.sin(beta-alpha))])
        metrics['net_force_fraction']=float(np.linalg.norm(net)/total) if total>0 else 1.
        metrics['torque_fraction']=float(abs(np.sum(force*np.sin(alpha)))/total) if total>0 else 1.
        thresholds['net_force_fraction']={'comparison':'<=','boundary':config['net_force_fraction_boundary']}
        thresholds['torque_fraction']={'comparison':'<=','boundary':config['torque_fraction_boundary']}
        if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Undefined metric')
        passed=all(v>=thresholds[k]['boundary'] if thresholds[k]['comparison']=='>=' else v<=thresholds[k]['boundary'] for k,v in metrics.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__': sys.exit(main())
