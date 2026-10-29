"""Standalone evaluator: never imports the solver or submitted Python code."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,25)

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

def observation_rms(model,raw):
    """Independent layered Maxwell transfer recursion for observation checking."""
    with np.load(raw,allow_pickle=False) as archive:
        period=archive['period'][0];z=archive['layer_top'][0]
        obs=np.r_[archive['log_apparent_resistivity'][0],archive['impedance_phase'][0]]
        std=np.r_[archive['log_resistivity_std'][0],archive['phase_std'][0]]
    omega=2*np.pi/period;mu=4*np.pi*1e-7
    with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
        resistivity=10.0**model[0]
        if np.any(resistivity==0):raise ValueError('Resistivity underflow')
        k=np.sqrt(1j*omega[:,None]*mu/resistivity[None,:]);c=1/k[:,-1]
        for j in range(z.size-2,-1,-1):
            v=-2*k[:,j]*(z[j+1]-z[j]);t=-np.expm1(v)/(1+np.exp(v))
            c=(k[:,j]*c+t)/(k[:,j]*(1+k[:,j]*c*t))
        predicted=np.r_[np.log10(np.abs(c)**2*omega*mu),np.arctan2(c.real,-c.imag)]
        result=float(np.sqrt(np.mean(((predicted-obs)/std)**2)))
    if not np.isfinite(result):raise ValueError('Non-finite observation metric')
    return result

def evaluate(prediction,asset_dir=HERE):
    """Return (JSON-ready report, exit code); paths are anchored at asset_dir."""
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        assets=Path(asset_dir)
        config=json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
        for key,comparison in [('roughness','<='),('observation_rms','<=')]:
            boundary=config[key+'_boundary']
            if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary):
                raise ValueError('Invalid metric boundary '+key)
            report['thresholds'][key]={'comparison':comparison,'boundary':boundary}
        a=load_array(prediction)
        residual=observation_rms(a,assets.parent/'data/raw_data.npz')
        roughness=float(np.sum(np.diff(a[0])**2))
        metrics={'roughness':roughness,'observation_rms':residual}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Non-finite metric')
        passed=roughness<=config['roughness_boundary'] and residual<=config['observation_rms_boundary']
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
