"""Standalone diffusion-amplitude and public attenuation evaluator."""
from pathlib import Path
import argparse,json,sys
import numpy as np

def load(path):
    with np.load(path,allow_pickle=False) as f:
        if 'reconstruction' not in f:raise ValueError('Missing reconstruction')
        x=f['reconstruction']
    if x.shape!=(1,29,256):raise ValueError('reconstruction must have shape (1,29,256)')
    if x.dtype!=np.float64:raise ValueError('reconstruction must have float64 dtype')
    if not np.all(np.isfinite(x)):raise ValueError('Nonfinite reconstruction')
    return x[0]

def score(pred,ref,K,Y,eta,cfg):
    norm=np.linalg.norm(ref,axis=1);pnorm=np.linalg.norm(pred,axis=1)
    if np.any(norm<=0):raise ValueError('Invalid zero reference norm')
    cosine=np.sum(pred*ref,axis=1)/np.maximum(pnorm*norm,np.finfo(float).tiny)
    metrics={'maximum_band_relative_l2':float(np.max(np.linalg.norm(pred-ref,axis=1)/norm)),
             'minimum_band_cosine':float(np.min(cosine)),
             'maximum_observation_ball_ratio':float(np.max(np.linalg.norm(pred@K.T-Y,axis=1)/eta)),
             'maximum_negative_relative_amplitude':float(np.max(np.maximum(-np.min(pred,axis=1),0)/np.maximum(np.max(abs(pred),axis=1),np.finfo(float).tiny)))}
    rules={'maximum_band_relative_l2':('<=',cfg['band_relative_l2_boundary']),'minimum_band_cosine':('>=',cfg['band_cosine_boundary']),
           'maximum_observation_ball_ratio':('<=',cfg['observation_ball_ratio_boundary']),'maximum_negative_relative_amplitude':('<=',cfg['negative_relative_amplitude_boundary'])}
    thresholds={k:{'comparison':direction,'boundary':bound} for k,(direction,bound) in rules.items()}
    if any(not np.isfinite(v['boundary']) or v['boundary']<=0 for v in thresholds.values()):raise ValueError('Invalid metric boundary')
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError('Nonfinite metric result (numerical overflow)')
    return metrics,thresholds

def hybrid_regularizer(z):
    z=np.maximum(z,0)
    return .99*np.sum(z,axis=-1)+.01*np.sum(np.where(z>0,z*np.log(np.maximum(z,np.finfo(float).tiny)),0),axis=-1)

def public_target_score(pred,K,Y,eta,cfg):
    regularizer=hybrid_regularizer(pred/Y[:,0,None])
    budget=np.asarray(cfg['regularizer_budgets'],dtype=float)
    if budget.shape!=(29,) or not np.isfinite(budget).all() or np.any(budget<=0):raise ValueError('Invalid regularizer budgets')
    metrics={'maximum_regularizer_budget_ratio':float(np.max(regularizer/budget)),
             'maximum_observation_ball_ratio':float(np.max(np.linalg.norm(pred@K.T-Y,axis=1)/eta)),
             'maximum_negative_relative_amplitude':float(np.max(np.maximum(-np.min(pred,axis=1),0)/np.maximum(np.max(abs(pred),axis=1),np.finfo(float).tiny)))}
    rules={'maximum_regularizer_budget_ratio':cfg['regularizer_budget_ratio_boundary'],'maximum_observation_ball_ratio':cfg['observation_ball_ratio_boundary'],'maximum_negative_relative_amplitude':cfg['negative_relative_amplitude_boundary']}
    if any(not np.isfinite(v) or v<=0 for v in rules.values()):raise ValueError('Invalid target metric boundary')
    if any(not np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric result (numerical overflow)')
    return metrics,{k:{'comparison':'<=','boundary':v} for k,v in rules.items()}

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]};code=2
    try:
        base=Path(__file__).resolve().parent;cfg=json.loads((base/'metrics.json').read_text(encoding='utf8'))
        with np.load(base.parent/'data/raw_data.npz',allow_pickle=False) as f:
            spectra=f['spectra'][0];support=f['support_indices'][0];b=f['b_values'][0]
        # Independent integration, noise calculation and physical forward model.
        breaks=np.r_[0,np.flatnonzero(np.diff(support)!=1)+1,len(support)]
        Y=np.stack([np.add.reduce(spectra[support[a:z]],axis=0) for a,z in zip(breaks[:-1],breaks[1:])])
        noise=np.median(np.std(spectra[:3000],axis=0,ddof=1))
        eta=np.sqrt((1.05*np.sqrt(30)*noise*np.sqrt(np.diff(breaks)))**2+(.001*np.linalg.norm(Y,axis=1))**2)
        D=np.geomspace(.1,50,256)*1e-10;K=np.exp(-b[:,None]*D[None,:])
        pred=load(Path(args.prediction));metrics,thresholds=public_target_score(pred,K,Y,eta,cfg)
        passed=all(metrics[k]<=v['boundary'] if v['comparison']=='<=' else metrics[k]>=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=thresholds);code=0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)]
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print(f'Report-write failure: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
