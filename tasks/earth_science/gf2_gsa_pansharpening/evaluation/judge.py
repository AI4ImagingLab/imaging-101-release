"""Independent deterministic evaluator. No solver imports or implicit alignment."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def metric_pair(estimate,reference):
    a=estimate.astype(np.float64).ravel();b=reference.astype(np.float64).ravel()
    norm=np.linalg.norm(a)*np.linalg.norm(b);span=float(np.ptp(b))
    if np.linalg.norm(b)==0 or span==0:raise ValueError('Undefined metrics for zero norm or constant reference')
    return (float(np.dot(a,b)/norm) if norm>0 else 0.0),float(np.sqrt(np.mean((a-b)**2))/span)
def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        contract=config['output_contract']
        if not np.isfinite(config['ncc_boundary']) or not -1<=config['ncc_boundary']<=1:
            raise ValueError('Invalid NCC metric boundary')
        if not np.isfinite(config['nrmse_boundary']) or config['nrmse_boundary']<0:
            raise ValueError('Invalid NRMSE metric boundary')
        # Asset/configuration failure is caught before scoring any prediction.
        refpath=assets/'ground_truth.npz'
        if not refpath.exists():refpath=assets/'baseline_reference.npz'
        with np.load(refpath,allow_pickle=False) as f:refs={k:f[k].copy() for k in contract}
        with np.load(Path(prediction),allow_pickle=False) as f:preds={k:f[k].copy() for k in contract}
        for key,entry in contract.items():
            for role,a in [('prediction',preds[key]),('reference',refs[key])]:
                if list(a.shape)!=entry['shape']:raise ValueError(f'{role} {key}: expected shape {entry["shape"]}, got {a.shape}')
                if a.dtype.name!=entry['dtype']:raise ValueError(f'{role} {key}: expected dtype {entry["dtype"]}, got {a.dtype}')
                if not np.all(np.isfinite(a)):raise ValueError(f'{role} {key}: NaN or infinity')
            ncc,nrmse=metric_pair(preds[key],refs[key])
            report['metrics'][key+'_ncc']=ncc;report['metrics'][key+'_nrmse']=nrmse
            report['thresholds'][key+'_ncc']={'comparison':'>=','boundary':config['ncc_boundary']}
            report['thresholds'][key+'_nrmse']={'comparison':'<=','boundary':config['nrmse_boundary']}
        passed=all(v>=report['thresholds'][k]['boundary'] if k.endswith('_ncc') else v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items())
        report['passed']=passed;report['status']='pass' if passed else 'fail'
        return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
