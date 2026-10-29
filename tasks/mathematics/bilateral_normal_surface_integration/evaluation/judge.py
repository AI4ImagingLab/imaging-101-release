"""Standalone evaluator: exact outputs, no implicit gauge fit or solver import."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ASSETS=Path(__file__).resolve().parent


def metrics(estimate,reference):
    """Uncentered cosine NCC, reference-dynamic-range RMS normalization."""
    r=reference.ravel();x=estimate.ravel();norm=np.linalg.norm(r);span=np.ptp(r)
    if norm<=0 or span<=0:raise ValueError('Degenerate evaluator reference')
    xnorm=np.linalg.norm(x)
    return float(np.dot(x,r)/(xnorm*norm)) if xnorm>0 else 0.,float(np.sqrt(np.mean((x-r)**2))/span)


def conditional_quality(height,normal,mask,k):
    """Independent signed one-sided normal operators and frozen-weight stationarity."""
    from scipy.sparse import coo_matrix,vstack
    from scipy.special import expit
    ids=np.full(mask.shape,-1,dtype=int);ids[mask]=np.arange(mask.sum());rr,cc=np.nonzero(mask);N=len(rr)
    nz=-normal[...,2];operators=[]
    for dr,dc,sign in [(-1,0,1),(1,0,-1),(0,1,1),(0,-1,-1)]:
        nr,nc=rr+dr,cc+dc;keep=(nr>=0)&(nr<mask.shape[0])&(nc>=0)&(nc<mask.shape[1])
        valid=np.flatnonzero(keep);valid=valid[mask[nr[valid],nc[valid]]]
        row=ids[rr[valid],cc[valid]];other=ids[nr[valid],nc[valid]];value=sign*nz[rr[valid],cc[valid]]
        operators.append(coo_matrix((np.r_[-value,value],(np.r_[row,row],np.r_[row,other])),shape=(N,N)).tocsr())
    up,down,right,left=operators;z=height[mask];b=np.concatenate([-normal[...,1][mask]]*2+[-normal[...,0][mask]]*2)
    vu,vd,hr,hl=[op@z for op in operators]
    wv=expit(k*(vd*vd-vu*vu));wh=expit(k*(hl*hl-hr*hr));w=np.concatenate([wv,1-wv,wh,1-wh]);matrix=vstack(operators).tocsr()
    residual=matrix@z-b;gradient=matrix.T@(w*residual);scale=np.linalg.norm(matrix.T@(w*b))
    stationarity=float(np.linalg.norm(gradient)/max(scale,1e-30));energy=float(residual@(w*residual))
    return energy,stationarity,float(abs(z.mean()))


def evaluate(prediction,asset_dir=ASSETS):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'));specs=config['outputs']
        if not specs or isinstance(config['energy_ratio_boundary'],bool) or not np.isfinite(config['energy_ratio_boundary']) or config['energy_ratio_boundary']<=0:raise ValueError('Invalid energy boundary')
        with np.load(assets/('ground_truth.npz' if (assets/'ground_truth.npz').exists() else 'baseline_reference.npz'),allow_pickle=False) as f:reference={k:f[k] for k in specs}
        with np.load(prediction,allow_pickle=False) as f:estimate={k:f[k] for k in specs}
        with np.load(assets.parent/'data/raw_data.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
        for label,arrays in [('prediction',estimate),('evaluator reference',reference)]:
            for key,spec in specs.items():
                a=arrays[key]
                if a.shape!=tuple(spec['shape']) or a.dtype!=np.dtype(spec['dtype']):raise ValueError('Invalid '+label+' shape/dtype: '+key)
                if not np.isfinite(a).all():raise ValueError('Nonfinite '+label+': '+key)
        values={};limits={}
        for name,key,k in [('vase','reconstruction',4.),('tent','tent_reconstruction',1.)]:
            mask=raw[name+'_mask'];normal=raw[name+'_normals']
            energy,stationarity,gauge=conditional_quality(estimate[key][0],normal,mask,k)
            refenergy,refstationarity,_=conditional_quality(reference[key][0],normal,mask,k)
            if refenergy<=0:raise ValueError('Degenerate standard normal residual')
            values[key+'_energy_ratio']=energy/refenergy;values[key+'_stationarity']=stationarity;values[key+'_gauge']=gauge
            values[key+'_outside']=float(np.max(abs(estimate[key][0][~mask]))) if (~mask).any() else 0.
            limits[key+'_energy_ratio']={'comparison':'<=','boundary':config['energy_ratio_boundary']}
            limits[key+'_stationarity']={'comparison':'<=','boundary':max(config['stationarity_floor'],config['stationarity_attainment_factor']*refstationarity)}
            limits[key+'_gauge']={'comparison':'<=','boundary':config['gauge_boundary']};limits[key+'_outside']={'comparison':'<=','boundary':0.}
        if not all(np.isfinite(v) for v in values.values()) or not all(np.isfinite(v['boundary']) and v['boundary']>=0 for v in limits.values()):raise ValueError('Undefined metric or boundary')
        passed=all(values[k]<=v['boundary'] for k,v in limits.items());report.update(status='pass' if passed else 'fail',passed=passed,metrics=values,thresholds=limits);return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code


if __name__=='__main__':sys.exit(main())
