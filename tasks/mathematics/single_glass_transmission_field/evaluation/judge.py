"""Independent reference and physical-stationarity Judge, no solver imports."""
from pathlib import Path
import argparse,json,sys,math
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,1080,1440,3)
def load_field(path,role):
    if not Path(path).is_file():raise ValueError(f'Missing {role}: {path}')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError(f'{role} requires exactly reconstruction')
        x=z['reconstruction']
    if x.shape!=SHAPE or x.dtype!=np.float64:raise ValueError(f'{role}: expected real float64 {SHAPE}')
    if not np.isfinite(x).all():raise ValueError(f'{role}: nonfinite values')
    return x[0]
def laplace(x):
    out=np.zeros_like(x);dy=x[1:]-x[:-1];dx=x[:,1:]-x[:,:-1]
    out[:-1]+=dy;out[1:]-=dy;out[:,:-1]+=dx;out[:,1:]-=dx
    return out
def reference_metrics(x,y):
    xr=x.ravel();yr=y.ravel();span=float(np.ptp(yr));ynorm=np.linalg.norm(yr);xnorm=np.linalg.norm(xr)
    if ynorm<=0 or span<=0 or not np.isfinite(ynorm):raise ValueError('Undefined reference metrics')
    return {'ncc':0. if xnorm==0 else float(xr@yr/xnorm/ynorm),'nrmse':float(np.sqrt(np.mean((x-y)**2))/span),'maximum_absolute':float(np.max(abs(x-y)))}
def physical_metrics(x,y):
    gx=np.zeros_like(y);gy=np.zeros_like(y);gx[:,:-1]=y[:,1:]-y[:,:-1];gy[:-1]=y[1:]-y[:-1]
    keep=np.sqrt(gx*gx+gy*gy)>.033;gx*=keep;gy*=keep
    target=gx+gy;target[:,1:]-=gx[:,:-1];target[1:]-=gy[:-1]
    rhs=laplace(target)+1e-8*y
    return {'stationarity_relative':float(np.linalg.norm(laplace(laplace(x))+1e-8*x-rhs)/np.linalg.norm(rhs)),'channel_mean_absolute':float(np.max(abs(x.mean(axis=(0,1))-y.mean(axis=(0,1)))))}
def evaluate(prediction,asset_dir=HERE):
    asset_dir=Path(asset_dir);thresholds={}
    config=json.loads((asset_dir/'metrics.json').read_text(encoding='utf8'))
    for name,direction in [('ncc','>='),('nrmse','<='),('maximum_absolute','<='),('stationarity_relative','<='),('channel_mean_absolute','<=')]:
        value=config[name+'_boundary']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (name=='ncc' and not -1<=value<=1) or (name!='ncc' and value<0):raise ValueError('Invalid metric threshold '+name)
        thresholds[name]={'comparison':direction,'boundary':float(value)}
    reference=asset_dir/'ground_truth.npz'
    if not reference.exists():reference=asset_dir/'baseline_reference.npz'
    yref=load_field(reference,'reference');reference_metrics(yref,yref)
    with np.load(asset_dir.parent/'data/raw_data.npz',allow_pickle=False) as z:
        if set(z.files)!={'observation'}:raise ValueError('Invalid evaluator observations')
        y=z['observation']
    if y.shape!=SHAPE or y.dtype!=np.float64 or not np.isfinite(y).all():raise ValueError('Invalid evaluator observation shape/dtype/values')
    x=load_field(prediction,'prediction');metrics=reference_metrics(x,yref);metrics.update(physical_metrics(x,y[0]))
    if not all(math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
    passed=all(metrics[k]>=v['boundary'] if v['comparison']=='>=' else metrics[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':metrics,'thresholds':thresholds,'errors':[]},0 if passed else 1
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args(argv)
    try:report,code=evaluate(a.prediction)
    except Exception as error:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(error)]};code=2
    try:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:print(f'Cannot write report: {error}',file=sys.stderr);return 2
    print(json.dumps({'status':report['status'],'report':str(a.report)}));return code
if __name__=='__main__':raise SystemExit(main())
