"""Standalone strict NPZ evaluator; no solver imports."""
import argparse,json,sys,importlib.util
from pathlib import Path
import numpy as np
ASSET_DIR=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('graphene_gradient_contract',ASSET_DIR/'gradient_contract.py')
_operator=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_operator)
residual_metrics=_operator.residual_metrics

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
        if not Path(prediction).is_file(): raise ValueError('Prediction file does not exist')
        with np.load(prediction,allow_pickle=False) as f:
            estimates={}
            for k,c in config['output_contract'].items():
                if k not in f: raise ValueError('Missing prediction key: '+k)
                v=f[k]
                if list(v.shape)!=c['shape']: raise ValueError('Wrong prediction shape: '+k)
                if v.dtype!=np.dtype(c['dtype']): raise ValueError('Wrong prediction dtype: '+k)
                if not np.isfinite(v).all(): raise ValueError('Nonfinite prediction: '+k)
                estimates[k]=v
        metrics=residual_metrics(estimates['reconstruction'][0],asset_dir.parent/'data/raw_data.npz')
        thresholds={k:{'comparison':'<=','boundary':config[k+'_boundary']} for k in metrics}
        if any(not np.isfinite(v['boundary']) or v['boundary']<0 for v in thresholds.values()):raise ValueError('Invalid metric boundary')
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
