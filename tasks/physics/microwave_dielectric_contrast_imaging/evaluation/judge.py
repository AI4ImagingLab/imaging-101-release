"""Independent NPZ Judge: actual cell-mean truth and physical real-field check."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.special import jv,hankel2
ROOT=Path(__file__).resolve().parent
SHAPE=(1,30,30)
def ncc(x,y):
    xn=np.linalg.norm(x);yn=np.linalg.norm(y)
    if yn==0:raise ValueError('Undefined zero reference cosine norm')
    return 0. if xn==0 else float(np.vdot(x.ravel(),y.ravel()).real/(xn*yn))
def nrmse(x,y):
    span=float(y.max()-y.min())
    if span<=0:raise ValueError('Undefined constant reference dynamic range')
    return float(np.sqrt(np.mean((x-y)**2))/span)
def physical_prediction(contrast,illumination_angles,receiver_angles):
    """Direct dense physical state inverse; independent of reference FFT/CG."""
    step=.8/30;coordinates=np.arange(-.4+.5*step,.4,step);x,y=np.meshgrid(coordinates,coordinates);x=x.ravel();y=y.ravel()
    a=np.sqrt((coordinates[1]-coordinates[0])**2/np.pi);k=12.575070131717595+0j
    distance=np.sqrt((x[:,None]-x[None,:])**2+(y[:,None]-y[None,:])**2);np.fill_diagonal(distance,1.)
    gd=-1j*np.pi*k*a/2*jv(1,k*a)*hankel2(0,k*distance);np.fill_diagonal(gd,-.5j*(np.pi*k*a*hankel2(1,k*a)-2j))
    rx=np.cos(receiver_angles);ry=np.sin(receiver_angles);distance=np.sqrt((rx[:,None]-x)**2+(ry[:,None]-y)**2)
    gs=-1j*np.pi*k*a/2*jv(1,k*a)*hankel2(0,k*distance)
    incident=np.exp(-1j*k*(x[:,None]*np.cos(illumination_angles)+y[:,None]*np.sin(illumination_angles)))
    # Keep the original real permittivity roundtrip, without any complex fit.
    chi=(4*(contrast.ravel()+1))/4-1
    field=np.linalg.solve(np.eye(900)-gd*chi[None,:],incident)
    return gs@(chi[:,None]*field)
def score_arrays(x,y):return {'ncc':ncc(x,y),'nrmse':nrmse(x,y)}
def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        definitions={'ncc':('>=',config['ncc_boundary']),'nrmse':('<=',config['nrmse_boundary']),'observation_residual':('<=',config['observation_residual_boundary'])}
        for key,(comparison,boundary) in definitions.items():
            if type(boundary) not in (int,float) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid metric boundary '+key)
            report['thresholds'][key]={'comparison':comparison,'boundary':boundary}
        ref_path=ROOT/'ground_truth.npz'
        if not ref_path.exists():ref_path=ROOT/'baseline_reference.npz'
        with np.load(ref_path,allow_pickle=False) as f:y=f['reconstruction'].copy()
        with np.load(prediction,allow_pickle=False) as f:x=f['reconstruction'].copy()
        for arr,role in [(x,'prediction'),(y,'reference')]:
            if arr.shape!=SHAPE:raise ValueError(role+' reconstruction shape must be '+str(SHAPE))
            if arr.dtype!=np.dtype('float64'):raise ValueError(role+' reconstruction dtype must be real float64')
            if not np.isfinite(arr).all():raise ValueError(role+' reconstruction contains NaN/Infinity')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            values=score_arrays(x,y)
            with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as f:
                data=f['measurements'][0];angles=f['illumination_angles'][0];receivers=f['receiver_angles'][0]
            if data.shape!=(9,10) or angles.shape!=(10,) or receivers.shape!=(9,) or not np.isfinite(data).all():raise ValueError('Invalid public acquisition calibration')
            denominator=np.linalg.norm(data)
            if denominator==0:raise ValueError('Undefined zero observation norm')
            predicted=physical_prediction(x[0],angles,receivers)
            values['observation_residual']=float(np.linalg.norm(predicted-data)/denominator)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite metric')
        passed=values['ncc']>=config['ncc_boundary'] and values['nrmse']<=config['nrmse_boundary'] and values['observation_residual']<=config['observation_residual_boundary']
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values)
        return report,0 if passed else 1
    except Exception as exc:report['metrics']={};report['errors']=[str(exc)];return report,2
def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args();report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
