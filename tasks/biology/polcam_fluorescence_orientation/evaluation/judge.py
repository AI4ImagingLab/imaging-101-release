"""Independent standalone evaluator. Never imports solver modules."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,116,3,149,149),'orientation':(1,116,149,149),'degree':(1,116,149,149)}

def ncc_nrmse(estimate,reference):
    e=estimate.ravel();r=reference.ravel();norm=float(np.linalg.norm(e)*np.linalg.norm(r)); span=float(np.max(r)-np.min(r))
    if norm==0 or span==0: raise ValueError('undefined NCC/NRMSE for degenerate arrays')
    return float(np.dot(e,r)/norm),float(np.sqrt(np.mean((e-r)**2))/span)

def evaluate(prediction: str | Path) -> tuple[dict,int]:
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        with np.load(prediction,allow_pickle=False) as f:
            estimate={}
            for key,shape in SHAPES.items():
                if key not in f:raise ValueError('missing required key '+key)
                a=f[key]
                if a.shape!=shape:raise ValueError('wrong shape for '+key)
                if a.dtype!=np.float64:raise ValueError('required float64 real dtype for '+key)
                if not np.isfinite(a).all():raise ValueError('nonfinite values in '+key)
                estimate[key]=a
        path=ROOT/'ground_truth.npz'
        if not path.exists():path=ROOT/'baseline_reference.npz'
        with np.load(path,allow_pickle=False) as f:
            reference={k:f[k] for k in SHAPES}
        for key,shape in SHAPES.items():
            if reference[key].shape!=shape or reference[key].dtype!=np.float64 or not np.isfinite(reference[key]).all():raise ValueError('invalid hidden reference '+key)
        cfg=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        measured={};thresholds={}
        for i in range(3):
            nc,nr=ncc_nrmse(estimate['reconstruction'][:,:,i],reference['reconstruction'][:,:,i])
            measured[f'stokes{i}_ncc']=nc;measured[f'stokes{i}_nrmse']=nr
            thresholds[f'stokes{i}_ncc']={'comparison':'>=','boundary':cfg['ncc_boundary']};thresholds[f'stokes{i}_nrmse']={'comparison':'<=','boundary':cfg['nrmse_boundary']}
        nc,nr=ncc_nrmse(estimate['degree'],reference['degree'])
        measured['degree_ncc']=nc;measured['degree_nrmse']=nr
        thresholds['degree_ncc']={'comparison':'>=','boundary':cfg['ncc_boundary']};thresholds['degree_nrmse']={'comparison':'<=','boundary':cfg['nrmse_boundary']}
        delta=estimate['orientation']-reference['orientation'];weight=np.sum(reference['reconstruction'][:,:,1:]**2,axis=2);den=float(weight.sum())
        if den<=0:raise ValueError('undefined angular metric')
        measured['angular_rmse']=float(np.sqrt(np.sum(weight*(np.arctan2(np.sin(2*delta),np.cos(2*delta))/2)**2)/den));thresholds['angular_rmse']={'comparison':'<=','boundary':cfg['angular_rmse_boundary']}
        with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as f:raw=f['intensity'].astype(np.float64)
        channels=np.stack((raw[...,::2,1::2],raw[...,1::2,1::2],raw[...,1::2,::2],raw[...,::2,::2]),axis=2)
        s0,s1,s2=np.moveaxis(estimate['reconstruction'],2,0);expected=np.stack(((s0+s1)/2,(s0+s2)/2,(s0-s1)/2,(s0-s2)/2),axis=2)
        measured['observation_residual']=float(np.linalg.norm((expected-channels).ravel())/np.linalg.norm(channels.ravel()));thresholds['observation_residual']={'comparison':'<=','boundary':cfg['observation_residual_boundary']}
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('undefined measured metric')
        passed=all(measured[k]>=v['boundary'] if v['comparison']=='>=' else measured[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds=thresholds)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}'];return report,2

def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    result,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
