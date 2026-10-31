"""Independent strict evaluator: no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
KEYS=('S0','diffusion','fraction','pseudodiffusion')

def load_prediction(path,mask):
    with np.load(path,allow_pickle=False) as z:
        values={}
        for k in KEYS:
            if k not in z:raise ValueError('Missing key '+k)
            v=z[k]
            if v.shape!=(1,128,134,6):raise ValueError('Invalid shape for '+k)
            if v.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Expected real float32/64 '+k)
            if not np.isfinite(v).all():raise ValueError('Nonfinite '+k)
            if np.any(v[0][~mask]!=0):raise ValueError('Nonzero outside public support '+k)
            values[k]=v[0][mask].astype(float)
    if np.any(values['S0']<=0):raise ValueError('S0 must be positive')
    for k,hi in [('diffusion',.01),('fraction',1),('pseudodiffusion',.1)]:
        if np.any((values[k]<0)|(values[k]>hi)):raise ValueError('Physical range '+k)
    return values

def metrics(p,r,b,observations):
    rms=lambda v:float(np.sqrt(np.mean(v*v)))
    wf=1-np.exp(-max(b)*r['pseudodiffusion']);wd=r['fraction']
    if np.linalg.norm(wf)==0 or np.linalg.norm(wd)==0:raise ValueError('Degenerate reference weights')
    S0,D,f,Ds=(p[k][:,None] for k in KEYS)
    pred=S0*np.exp(-b*D)*(f*np.exp(-b*Ds)+1-f)
    norm=np.linalg.norm(observations)
    if norm==0:raise ValueError('Zero observation norm')
    return {'S0_rmse':rms(p['S0']-r['S0']),'D_rmse':rms(p['diffusion']-r['diffusion']),'fraction_weighted_rmse':float(np.linalg.norm(wf*(p['fraction']-r['fraction']))/np.linalg.norm(wf)),'Dstar_weighted_rmse':float(np.linalg.norm(wd*(p['pseudodiffusion']-r['pseudodiffusion']))/np.linalg.norm(wd)),'fDstar_rmse':rms(p['fraction']*p['pseudodiffusion']-r['fraction']*r['pseudodiffusion']),'observation_relative_l2':float(np.linalg.norm(pred-observations)/norm)}

def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf-8'))
        with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as raw:
            mask=raw['observation_mask'][0];b=raw['bvalues'][0];obs=raw['measurements'][0][mask]
        reference=ROOT/'ground_truth.npz'
        if not reference.exists():reference=ROOT/'baseline_reference.npz'
        r=load_prediction(reference,mask);p=load_prediction(prediction,mask)
        measured=metrics(p,r,b,obs);thresholds={}
        for name in measured:
            boundary=config[name+'_boundary']
            if not isinstance(boundary,(float,int)) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid metric boundary '+name)
            thresholds[name]={'comparison':'<=','boundary':boundary}
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Undefined metrics')
        passed=all(v<=thresholds[k]['boundary'] for k,v in measured.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
