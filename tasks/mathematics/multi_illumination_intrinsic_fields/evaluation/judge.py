"""Standalone evaluator for two jointly required intrinsic fields. No solver imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
FIELDS=('shading','reflectance')
def aligned_metrics(estimate,reference,mask):
    x=estimate[mask];y=reference[mask];xx=float(x@x);yy=float(y@y);span=float(np.ptp(y))
    if xx<=0 or yy<=0 or span<=0:raise ValueError('Undefined reference or estimate metric')
    scale=max(0.,float(x@y)/xx)
    return {'ncc':float(x@y/np.sqrt(xx)/np.sqrt(yy)),'nrmse':float(np.sqrt(np.mean((scale*x-y)**2))/span),'positive_scale':scale,'source_lmse':local_error(reference,estimate,mask)}
def local_error(reference,estimate,mask,window=20,step=10):
    total=0.;error=0.;height,width=reference.shape
    for i in range(0,height-window+1,step):
        for j in range(0,width-window+1,step):
            m=mask[i:i+window,j:j+window];x=estimate[i:i+window,j:j+window];y=reference[i:i+window,j:j+window]
            xx=float(np.sum(x*x*m));scale=float(np.sum(x*y*m)/xx) if xx>1e-5 else 0.
            error+=float(np.sum(m*(y-scale*x)**2));total+=float(np.sum(m*y*y))
    if total<=0:raise ValueError('Undefined local reference energy')
    return error/total
def numeric(value,name,lo=0.,hi=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<lo or (hi is not None and value>hi):raise ValueError(f'Invalid threshold {name}')
    return float(value)
def evaluate(prediction):
    with (HERE/'metrics.json').open(encoding='utf8') as f:config=json.load(f)
    thresholds={}
    for field in FIELDS:
        for key,direction in [('ncc','>= '),('nrmse','<= '),('source_lmse','<= ')]:
            boundary=numeric(config[field][key+'_boundary'],field+'_'+key,lo=-1. if key=='ncc' else 0.,hi=1. if key=='ncc' else None)
            thresholds[field+'_'+key]={'comparison':direction.strip(),'boundary':boundary}
    for key in ['product_relative_l2','gauge_absolute_log_mean','outside_absolute']:
        thresholds[key]={'comparison':'<=','boundary':numeric(config[key+'_boundary'],key)}
    if not prediction.is_file():raise ValueError(f'Prediction file missing: {prediction}')
    with np.load(prediction,allow_pickle=False) as z:
        if set(z.files)!=set(FIELDS):raise ValueError('Prediction must contain exactly shading and reflectance')
        submission={key:z[key] for key in FIELDS}
    for key,a in submission.items():
        if a.shape!=(1,600,534):raise ValueError(f'{key}: expected shape (1,600,534)')
        if a.dtype!=np.dtype('float64'):raise ValueError(f'{key}: expected real float64')
        if not np.isfinite(a).all():raise ValueError(f'{key}: nonfinite values')
    with np.load(HERE/'ground_truth.npz',allow_pickle=False) as z:truth={key:z[key] for key in FIELDS}
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:mask=z['object_mask'][0];image=z['reference_image'][0]
    if mask.shape!=(600,534) or mask.dtype!=np.bool_ or not mask.any() or image.shape!=mask.shape or not np.isfinite(image).all():raise ValueError('Invalid evaluator calibration')
    for key,a in truth.items():
        if a.shape!=(1,600,534) or a.dtype!=np.float64 or not np.isfinite(a).all() or np.linalg.norm(a[0][mask])==0 or np.ptp(a[0][mask])==0:raise ValueError('Invalid hidden independent reference')
    if any(np.any(a[0][mask]<=0) for a in submission.values()):raise ValueError('Fields must be strictly positive on foreground')
    metrics={}
    for key in FIELDS:
        result=aligned_metrics(submission[key][0],truth[key][0],mask)
        metrics.update({key+'_'+k:v for k,v in result.items()})
    metrics['product_relative_l2']=float(np.linalg.norm((submission['shading'][0]*submission['reflectance'][0]-image)[mask])/np.linalg.norm(image[mask]))
    metrics['gauge_absolute_log_mean']=float(abs(np.log(submission['reflectance'][0][mask]).mean()))
    metrics['outside_absolute']=max(float(np.max(abs(a[0][~mask]))) for a in submission.values())
    if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
    passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':metrics,'thresholds':thresholds,'errors':[]},0 if passed else 1
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args(argv)
    try:report,code=evaluate(args.prediction)
    except Exception as error:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(error)]};code=2
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:print(f'Cannot write report: {error}',file=sys.stderr);return 2
    print(json.dumps({'status':report['status'],'report':str(args.report)}));return code
if __name__=='__main__':raise SystemExit(main())
