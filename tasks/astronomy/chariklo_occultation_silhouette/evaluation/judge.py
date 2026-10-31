"""Independent ray/ellipse contact evaluator; no reconstruction imports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PARAMETERS=('center_f_error','center_g_error','radius_error','oblateness_error','position_angle_error')
METRICS=PARAMETERS+('contour_ncc','contour_nrmse','contour_consistency_km','timing_sigma_rms','missing_contact_count','negative_penetration_km')

def read_output(path):
    with np.load(path,allow_pickle=False) as z:out={k:z[k].copy() for k in ('parameters','reconstruction')}
    for k,shape in [('parameters',(1,5)),('reconstruction',(1,360,2))]:
        a=out[k]
        if a.shape!=shape or a.dtype!=np.dtype('float64') or np.iscomplexobj(a) or not np.isfinite(a).all():raise ValueError(k+' must be finite real float64 '+str(shape))
    p=out['parameters'][0]
    if not(p[2]>0 and 0<=p[3]<1):raise ValueError('invalid ellipse radius/oblateness')
    return out

def rotation(parameters):
    angle=np.deg2rad(parameters[4]);c=np.cos(angle);s=np.sin(angle)
    return np.array([[c,-s],[s,c]])

def independent_contour(parameters,theta):
    # Ellipse quadratic form in fixed sky axes, not solver get_ellipse_points.
    r=rotation(parameters);axes=np.array([parameters[2],parameters[2]*(1-parameters[3])])
    q=r.T@np.diag(1/axes**2)@r
    direction=np.column_stack([np.cos(theta),np.sin(theta)])
    radii=1/np.sqrt(np.einsum('ni,ij,nj->n',direction,q,direction))
    return parameters[:2]+direction*radii[:,None]

def observation_metrics(parameters,raw):
    p=parameters;r=rotation(p);axes=np.array([p[2],p[2]*(1-p[3])]);q=r.T@np.diag(1/axes**2)@r
    x=raw['limb_points']-p[:2];v=raw['velocity']
    a=np.einsum('ni,ij,nj->n',v,q,v);b=2*np.einsum('ni,ij,nj->n',x,q,v);c=np.einsum('ni,ij,nj->n',x,q,x)-1
    disc=b*b-4*a*c
    misses=int(np.count_nonzero(disc<0))
    # A missed ray is a valid poor geometry, and separately fails the mandatory
    # zero-missing-contact criterion. Its quadratic vertex supplies a finite
    # time statistic without pretending an intersection occurred.
    roots=np.column_stack([(-b-np.sqrt(np.maximum(disc,0)))/(2*a),(-b+np.sqrt(np.maximum(disc,0)))/(2*a)])
    timing=np.min(abs(roots),axis=1)
    rms=float(np.sqrt(np.mean((timing/raw['timing_sigma'])**2)))
    delta=raw['negative_path']-p[:2];directions=delta/np.linalg.norm(delta,axis=1)[:,None]
    limb=1/np.sqrt(np.einsum('ni,ij,nj->n',directions,q,directions))
    penetration=float(max(0,np.max(limb-np.linalg.norm(delta,axis=1))))
    return rms,misses,penetration

def compare(a,b):
    a=a.ravel();b=b.ravel();norm=np.linalg.norm(a)*np.linalg.norm(b);dynamic=np.ptp(b)
    if not(norm>0 and dynamic>0):raise ValueError('degenerate contour/reference')
    return float(a@b/norm),float(np.sqrt(np.mean(((a-b)/dynamic)**2)))

def evaluate(prediction):
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));thresholds={}
    for key in METRICS:
        boundary=config[key+'_boundary'];cosine=key=='contour_ncc'
        if isinstance(boundary,bool) or not isinstance(boundary,(int,float)) or not np.isfinite(boundary) or (cosine and not -1<=boundary<=1) or (not cosine and boundary<0):raise ValueError('invalid boundary '+key)
        thresholds[key]={'comparison':'>=' if cosine else '<=','boundary':boundary}
    estimate=read_output(prediction);reference=HERE/'ground_truth.npz'
    if not reference.exists():reference=HERE/'baseline_reference.npz'
    reference=read_output(reference);p=estimate['parameters'][0];b=reference['parameters'][0]
    error=abs(p-b);error[4]=abs((p[4]-b[4]+90)%180-90)
    values={key:float(value) for key,value in zip(PARAMETERS,error)}
    values['contour_ncc'],values['contour_nrmse']=compare(estimate['reconstruction'],reference['reconstruction'])
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k][0] for k in z.files}
    expected=independent_contour(p,raw['contour_angles'])
    values['contour_consistency_km']=float(np.sqrt(np.mean((estimate['reconstruction'][0]-expected)**2)))
    values['timing_sigma_rms'],values['missing_contact_count'],values['negative_penetration_km']=observation_metrics(p,raw)
    if not np.isfinite(list(values.values())).all():raise ValueError('nonfinite computed metrics')
    passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':passed,'metrics':values,'thresholds':thresholds,'errors':[]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    try:report=evaluate(args.prediction);code=0 if report['passed'] else 1
    except Exception as exc:report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(exc)]};code=2
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
