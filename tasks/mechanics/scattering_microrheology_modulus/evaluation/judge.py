"""Standalone evaluator. Never imports solver or submitted Python."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,488,2),'msd':(1,488),'exponent':(1,488),'intercept':(1,3)}
LENGTHS=(188,145,155)
def load(path):
    if not path.is_file():raise ValueError(f'Missing NPZ: {path}')
    with np.load(path,allow_pickle=False) as z:
        result={}
        for k,shape in SHAPES.items():
            if k not in z:raise ValueError(f'Missing required key: {k}')
            a=z[k]
            if a.shape!=shape:raise ValueError(f'{k}: expected shape {shape}, got {a.shape}')
            if a.dtype!=np.dtype('float64'):raise ValueError(f'{k}: expected real float64')
            if not np.isfinite(a).all():raise ValueError(f'{k}: nonfinite values')
            result[k]=a
    return result
def agreement(a,b):
    a,b=a.ravel(),b.ravel();scale=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if scale<=0 or norm<=0:raise ValueError('Evaluator reference is degenerate')
    na=np.linalg.norm(a)
    return float(a@b/(na*norm)) if na else 0.,float(np.sqrt(np.mean((a-b)**2))/scale)
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        for k in ('ncc_boundary','nrmse_boundary','intercept_relative_error_boundary','forward_rmse_boundary'):
            if not np.isfinite(config[k]):raise ValueError(f'Invalid evaluator boundary {k}')
        reference=HERE/'ground_truth.npz'
        if not reference.exists():reference=HERE/'baseline_reference.npz'
        ref=load(reference);pred=load(Path(prediction))
        measures={};bounds={}
        pairs=[('storage',pred['reconstruction'][...,0],ref['reconstruction'][...,0]),('loss',pred['reconstruction'][...,1],ref['reconstruction'][...,1]),('msd',pred['msd'],ref['msd']),('exponent',pred['exponent'],ref['exponent'])]
        for name,a,b in pairs:
            nc,nr=agreement(a,b)
            measures[name+'_ncc']=nc;bounds[name+'_ncc']={'comparison':'>=','boundary':config['ncc_boundary']}
            measures[name+'_nrmse']=nr;bounds[name+'_nrmse']={'comparison':'<=','boundary':config['nrmse_boundary']}
        measures['intercept_relative_error']=float(np.max(abs(pred['intercept']/ref['intercept']-1)))
        bounds['intercept_relative_error']={'comparison':'<=','boundary':config['intercept_relative_error_boundary']}
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:g=z['correlation'][0]
        q=4*np.pi*1.333*np.sin(np.deg2rad(173)/2)/633.
        errors=[];start=0
        with np.errstate(over='raise',invalid='raise'):
            for i,n in enumerate(LENGTHS):
                fit=pred['intercept'][0,i]*np.exp(-q*q*pred['msd'][0,start:start+n]/3)
                errors.append(np.sqrt(np.mean((fit-g[i,3:3+n])**2)));start+=n
        measures['forward_rmse']=float(max(errors))
        bounds['forward_rmse']={'comparison':'<=','boundary':config['forward_rmse_boundary']}
        if not np.isfinite(list(measures.values())).all():raise ValueError('Nonfinite evaluation metric')
        passed=all(v>=bounds[k]['boundary'] if bounds[k]['comparison']=='>=' else v<=bounds[k]['boundary'] for k,v in measures.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measures,thresholds=bounds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
