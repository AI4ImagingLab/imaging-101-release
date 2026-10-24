"""Independent standalone spectral-field evaluator, no solver or submitted imports."""
from pathlib import Path
import json,argparse,sys
import numpy as np
SHAPES={'reconstruction':(1,76,2,1,111,111),'saturation':(1,76,1,111,111)}
def metrics(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();span=np.ptp(b);na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if span<=0 or nb<=0:raise ValueError('Degenerate reference')
    return dict(ncc=float(a@b/(na*nb)) if na else 0.,nrmse=float(np.sqrt(np.mean((a-b)**2))/span))

def spectral_residual(coefficient,observed,spectra,support):
    predicted=np.einsum('tczyx,cw->twzyx',coefficient.astype(np.float64),spectra)
    observed=observed.astype(np.float64)
    return float(np.linalg.norm((predicted-observed)[...,support])/np.linalg.norm(observed[...,support]))

def evaluate(prediction,asset_dir=None):
    base=Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((base/'metrics.json').read_text(encoding='utf-8'))
        for k in ['ncc_boundary','nrmse_boundary','spectral_relative_residual_boundary']:
            if not isinstance(config.get(k),(int,float)) or not np.isfinite(config[k]):raise ValueError('Invalid metric config '+k)
        if not 0<config['ncc_boundary']<=1 or config['nrmse_boundary']<=0 or config['spectral_relative_residual_boundary']<=0:raise ValueError('Invalid metric boundaries')
        reference_path=base/('ground_truth.npz' if (base/'ground_truth.npz').exists() else 'baseline_reference.npz')
        with np.load(reference_path,allow_pickle=False) as z:reference={k:z[k] for k in z.files}
        if set(reference)!=set(SHAPES):raise ValueError('Invalid reference keys')
        with np.load(Path(__file__).resolve().parent.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        support=raw['roi_labels'][0]>0
        with np.load(Path(prediction),allow_pickle=False) as z:submission={k:z[k] for k in z.files}
        if set(submission)!=set(SHAPES):raise ValueError('Require exactly reconstruction and saturation')
        for k,shape in SHAPES.items():
            for role,arrays in [('reference',reference),('prediction',submission)]:
                a=arrays[k]
                if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Invalid '+role+' '+k+' shape/dtype/finiteness')
            if np.any(submission[k][...,~support]!=0):raise ValueError('Nonzero exterior '+k)
        values={};thresholds={}
        for name,a,b in [('Hb',submission['reconstruction'][0,:,0],reference['reconstruction'][0,:,0]),('HbO2',submission['reconstruction'][0,:,1],reference['reconstruction'][0,:,1]),('saturation',submission['saturation'][0],reference['saturation'][0])]:
            for metric,value in metrics(a[...,support],b[...,support]).items():
                key=name+'_'+metric;values[key]=value;thresholds[key]=dict(comparison='>=' if metric=='ncc' else '<=',boundary=config[metric+'_boundary'])
        values['spectral_relative_residual']=spectral_residual(submission['reconstruction'][0],raw['images'][0],raw['spectra'][0],support)
        thresholds['spectral_relative_residual']=dict(comparison='<=',boundary=config['spectral_relative_residual_boundary'])
        if not all(np.isfinite(x) for x in values.values()):raise ValueError('Nonfinite metric')
        passed=all(v>=thresholds[k]['boundary'] if thresholds[k]['comparison']=='>=' else v<=thresholds[k]['boundary'] for k,v in values.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=thresholds)
    except Exception as e:report['errors']=[str(e) or type(e).__name__]
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
