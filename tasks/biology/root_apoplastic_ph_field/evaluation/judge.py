"""Independent imagewise pH evaluation; never imports submitted code."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,43,1024,1024)


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
        for key in ['ncc_boundary','nrmse_boundary']:
            v=config[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Undefined metric boundary')
        if not -1<=config['ncc_boundary']<=1 or config['nrmse_boundary']<0:raise ValueError('Invalid metric boundary')
        with np.load(HERE/'baseline_reference.npz',allow_pickle=False) as a:reference=a['reconstruction']
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as a:ratio=a['ratio']
        if reference.shape!=SHAPE or reference.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(reference).all():raise ValueError('Invalid reference asset')
        if ratio.shape!=SHAPE or ratio.dtype!=np.float32 or np.isinf(ratio).any():raise ValueError('Invalid public observation asset')
        mask=np.isfinite(ratio)&(ratio!=0)
        del ratio
        for image,valid in zip(reference[0],mask[0]):
            y=image[valid].astype(float)
            if not len(y) or np.linalg.norm(y)==0 or np.ptp(y)==0:raise ValueError('Degenerate image reference')
        if np.any(reference[~mask]!=0):raise ValueError('Reference padding disagrees with public support')
    except Exception as exc:
        report['errors']=['Evaluator error: '+str(exc)];return report
    report['thresholds']={'minimum_image_ncc':{'comparison':'>=','boundary':config['ncc_boundary']},'maximum_image_nrmse':{'comparison':'<=','boundary':config['nrmse_boundary']}}
    try:
        path=Path(prediction)
        if not path.is_file():raise ValueError('Prediction file does not exist')
        with np.load(path,allow_pickle=False) as a:
            if set(a.files)!={'reconstruction'}:raise ValueError('Require exactly reconstruction key')
            estimate=a['reconstruction']
        if estimate.shape!=SHAPE:raise ValueError('Wrong prediction shape; no crop/reshape allowed')
        if estimate.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Require real float32 or float64')
        if not np.isfinite(estimate).all():raise ValueError('Nonfinite prediction')
        if np.any(estimate[~mask]!=0):raise ValueError('Missing-observation pixels must be exactly zero')
    except Exception as exc:
        report['errors']=['Invalid prediction: '+str(exc)];return report
    metrics={};nccs=[];nrmses=[]
    for i,(image,ref,valid) in enumerate(zip(estimate[0],reference[0],mask[0])):
        x=image[valid].astype(float);y=ref[valid].astype(float)
        norm=np.linalg.norm(x)
        ncc=float(np.dot(x,y)/norm/np.linalg.norm(y)) if norm else 0.
        nrmse=float(np.sqrt(np.mean((x-y)**2))/np.ptp(y))
        if not np.isfinite([ncc,nrmse]).all():report['errors']=['Invalid prediction: metric overflow'];return report
        metrics[f'image_{i:02d}_ncc']=ncc;metrics[f'image_{i:02d}_nrmse']=nrmse;nccs.append(ncc);nrmses.append(nrmse)
    metrics['minimum_image_ncc']=min(nccs);metrics['maximum_image_nrmse']=max(nrmses)
    passed=min(nccs)>=config['ncc_boundary'] and max(nrmses)<=config['nrmse_boundary']
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Could not write report: '+str(exc),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2


if __name__=='__main__':raise SystemExit(main())
