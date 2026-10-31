"""Independent NPZ evaluator; never imports submitted or reference solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def load(path):
    with np.load(path,allow_pickle=False) as f:
        arrays={}
        for k,shape in [('reconstruction',(1,3)),('source_blend_flux',(1,2))]:
            if k not in f:raise ValueError('Missing key: '+k)
            a=f[k]
            if a.shape!=shape or a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Invalid shape/dtype: '+k)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+k)
            arrays[k]=a
    return arrays
def score(a,b,data):
    p=a['reconstruction'][0];r=b['reconstruction'][0];f=a['source_blend_flux'][0];rf=b['source_blend_flux'][0]
    if p[1]<=0 or p[2]<=0 or f[0]<=0:raise ValueError('Physical contract requires positive u0,tE,source flux')
    t=data['time'][0];m=data['magnitude'][0];me=data['magnitude_error'][0]
    observed=10.**(.4*(22.-m));error=me*observed*np.log(10.)*.4
    u=np.sqrt(((t-p[0])/p[2])**2+p[1]**2);model=f[0]*(u*u+2)/(u*np.sqrt(u*u+4))+f[1]
    return dict(t0_error=float(abs(p[0]-r[0])),u0_error=float(abs(p[1]-r[1])),tE_error=float(abs(p[2]-r[2])),flux_error=float(np.linalg.norm(f-rf)/rf[0]),reduced_chi2=float(np.sum(((observed-model)/error)**2)/(len(t)-5)))
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[]);code=2
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        for k in ['t0_error','u0_error','tE_error','flux_error','reduced_chi2']:
            boundary=config[k+'_boundary']
            if not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid metric boundary')
            report['thresholds'][k]={'comparison':'<=','boundary':boundary}
        ref=HERE/'ground_truth.npz'
        if not ref.exists():ref=HERE/'baseline_reference.npz'
        b=load(ref);r=b['reconstruction'][0];rf=b['source_blend_flux'][0]
        if r[1]<=0 or r[2]<=0 or rf[0]<=0:raise ValueError('Invalid hidden reference physicality')
        a=load(args.prediction)
        with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
        metrics=score(a,b,data)
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Undefined metric')
        report['metrics']=metrics;report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in metrics.items());report['status']='pass' if report['passed'] else 'fail';code=0 if report['passed'] else 1
    except Exception as exc:report['errors']=[str(exc)];report['metrics']={}
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
