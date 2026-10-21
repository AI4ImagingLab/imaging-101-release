"""Standalone evaluator: never imports the solver or submitted Python code."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,25,64,48)

def load_array(path):
    """Validate the complete declared submission/reference array contract."""
    if not Path(path).is_file():raise ValueError(f'File not found: {path}')
    with np.load(path,allow_pickle=False) as archive:
        if 'reconstruction' not in archive:raise ValueError('Missing required key reconstruction')
        a=archive['reconstruction'].copy()
    if a.shape!=SHAPE:raise ValueError(f'Expected shape {SHAPE}, got {a.shape}')
    if a.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 reconstruction')
    if not np.isfinite(a).all():raise ValueError('Non-finite reconstruction')
    return a

def similarity(a,b):
    """Cosine NCC and RMS error divided by reference dynamic range."""
    x=a.ravel();y=b.ravel()
    # Scaling keeps valid large finite submissions from overflowing norms/dots.
    sx=float(np.max(np.abs(x)));sy=float(np.max(np.abs(y)))
    if sy==0 or np.ptp(y)==0:raise ValueError('Degenerate hidden reference')
    ncc=0.0 if sx==0 else float(np.dot(x/sx,y/sy)/(np.linalg.norm(x/sx)*np.linalg.norm(y/sy)))
    with np.errstate(over='raise',invalid='raise'):
        nrmse=float(np.sqrt(np.mean(((x-y)/np.ptp(y))**2)))
    return ncc,nrmse

def balance_nrmse(model,raw):
    """Independent thin-plate energy conservation residual over every target pixel."""
    with np.load(raw,allow_pickle=False) as archive:
        t=archive['temperature'][0]
    c=t[:,1:-1,1:-1]
    dt=np.empty_like(t)
    dt[1:-1]=(t[2:]-t[:-2])/2
    dt[0]=(-3*t[0]+4*t[1]-t[2])/2
    dt[-1]=(3*t[-1]-4*t[-2]+t[-3])/2
    curvature=(t[:,2:,1:-1]+t[:,:-2,1:-1]+t[:,1:-1,2:]+t[:,1:-1,:-2]-4*c)/.0127**2
    conductivity=8.1119*np.log(c)-30.377
    heatcapacity=1000*(.1414*np.log(c)-.3534)
    storage=heatcapacity*8050*.000793*dt[:,1:-1,1:-1]
    loss=20*(c-293.15)+.94*5.67e-8*(2*c**4-293.15**4)
    rhs=storage+loss-conductivity*.000793*curvature
    with np.errstate(over='raise',invalid='raise'):
        score=float(np.sqrt(np.mean(((.94*1000*model[0]-rhs)/np.sqrt(np.mean(rhs**2)))**2)))
    if not np.isfinite(score):raise ValueError('Non-finite balance metric')
    return score


def evaluate(prediction,asset_dir=HERE):
    """Return (JSON-ready report, exit code); paths are anchored at asset_dir."""
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir)
        config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        for key,comparison in [('ncc','>='),('nrmse','<='),('balance_nrmse','<=')]:
            boundary=config[key+'_boundary']
            if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):
                raise ValueError('Invalid metric boundary '+key)
            report['thresholds'][key]={'comparison':comparison,'boundary':boundary}
        refpath=assets/'ground_truth.npz'
        if not refpath.exists():refpath=assets/'baseline_reference.npz'
        ref=load_array(refpath);a=load_array(prediction)
        ncc,nrmse=similarity(a,ref)
        residual=balance_nrmse(a,assets.parent/'data/raw_data.npz')
        metrics={'ncc':ncc,'nrmse':nrmse,'balance_nrmse':residual}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Non-finite metric')
        passed=ncc>=config['ncc_boundary'] and nrmse<=config['nrmse_boundary'] and residual<=config['balance_nrmse_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=metrics)
        return report,0 if passed else 1
    except Exception as error:
        report['metrics']={};report['errors']=[str(error)]
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args(argv);report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as error:
        print('Unable to write report: '+str(error),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
