"""Independent spherical forward check and strict NPZ evaluator."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
METRICS=('density_ncc','density_nrmse','emissivity_ncc','emissivity_nrmse','density_observation_rms','emissivity_observation_rms')

def compare(a,b):
    dynamic=np.ptp(b)
    if not dynamic>0:raise ValueError('constant evaluator reference')
    x=a.ravel();y=b.ravel();scale=max(float(np.max(np.abs(x))),1.)
    norm=np.linalg.norm(x/scale)
    cosine=0. if norm==0 else float((x/scale)@y/(norm*np.linalg.norm(y)))
    error=float(np.sqrt(np.mean(((a-b)/dynamic)**2)))
    if not np.isfinite([cosine,error]).all():raise ValueError('nonfinite metric arithmetic')
    return cosine,error

def observed_profile(raw_path):
    with np.load(raw_path,allow_pickle=False) as z:
        counts=z['counts'][0];exp=z['exposure'][0];bkg=z['instrument_background'][0];edges=z['radial_edges'][0]
    y,x=np.indices(counts.shape);distance=np.hypot(x-271.,y-280.)*.2491218
    valid=exp>.05*exp.max()
    def extract(e):
        rates=[];errors=[]
        for i in range(len(e)-1):
            mask=valid&(distance<np.round(e[i+1],5)+.000005)&(distance>=(0. if i==0 else np.round(e[i],5)+.000005))
            area=mask.sum()*.2491218**2
            rates.append((np.sum(counts[mask]/exp[mask])-np.sum(bkg[mask]/exp[mask]))/area)
            errors.append(np.sqrt(np.sum(counts[mask]/exp[mask]**2))/area)
        return np.asarray(rates),np.asarray(errors)
    oe=np.arange(136)/3.;op,err=extract(oe);centers=(oe[:-1]+oe[1:])/2
    mask=(centers>=30)&(centers<=40)&(err>0);weights=1/err[mask]**2
    background=np.sum(op[mask]*weights)/weights.sum();background_variance=1/weights.sum()
    data,sigma=extract(edges)
    return data-background,np.sqrt(sigma**2+background_variance),edges

def observation_rms(emissivity,raw_path):
    """Independent cylinder-intersection volume and count conversion."""
    observed,sigma,edges=observed_profile(raw_path)
    inner=edges[:-1];outer=edges[1:]
    annulus_area=np.pi*(outer**2-inner**2)
    # Geometric form differs from the normalized-ratio source implementation.
    # Volumes are first evaluated in angular units, then converted to cm3.
    pos=lambda a:np.maximum(a,0)**1.5
    ri=inner[:,None];ro=outer[:,None];si=inner[None,:];so=outer[None,:]
    v=4*np.pi/3*(pos(so**2-ri**2)-pos(so**2-ro**2)-pos(si**2-ri**2)+pos(si**2-ro**2))
    v*=(70.73126033926604*3.0856776e21)**3
    # Independent use of angular-diameter distance and the XSPEC definition.
    da_cm=272.69501551950657*3.0856776e24/(1.059**2)
    norm_per_em=1e-14/(4*np.pi*da_cm**2*1.059**2)
    source_rate=(v@emissivity)*(1/.8527)*norm_per_em*43.53/annulus_area
    # McLaughlin fixed exterior contribution: the observed outer annulus sets
    # its amplitude. This uses public observations, not a reference atmosphere.
    a=inner[-1];b=outer[-1]
    tail=(a+b)*a*b/outer**3*(-1+2/np.pi*(2*outer/b+np.arccos(outer/b)-outer/b*np.sqrt(1-outer**2/b**2)))
    mask=inner!=0;u=inner[mask];w=outer[mask]
    tail[mask]=(a+b)*a*b/(u+w)/u/w*(1-2/np.pi*((w/u*np.arccos(u/b)-np.arccos(w/b))-w/b*(np.sqrt(1-u*u/b**2)-np.sqrt(1-w*w/b**2)))/(w/u-1))
    prediction=source_rate+tail*observed[-1]
    result=float(np.sqrt(np.mean(((prediction-observed)/sigma)**2)))
    if not np.isfinite(result):raise ValueError('nonfinite observation RMS')
    return result

def read_output(path):
    with np.load(path,allow_pickle=False) as z:
        out={k:z[k].copy() for k in ('reconstruction','emissivity')}
    for key,value in out.items():
        if value.shape!=(1,35) or value.dtype!=np.dtype('float64') or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError(key+' must be finite real float64 (1,35)')
    return out

def evaluate(prediction):
    config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'));thresholds={}
    for key in METRICS:
        value=config[key+'_boundary'];cosine=key.endswith('_ncc')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or (not cosine and value<0) or (cosine and not -1<=value<=1):
            raise ValueError('invalid metric boundary: '+key)
        thresholds[key]={'comparison':'>=' if cosine else '<=','boundary':value}
    a=read_output(prediction);reference=HERE/'ground_truth.npz'
    if not reference.exists():reference=HERE/'baseline_reference.npz'
    b=read_output(reference);values={}
    for public,name in [('reconstruction','density'),('emissivity','emissivity')]:
        ncc,nrmse=compare(a[public],b[public]);values[name+'_ncc']=ncc;values[name+'_nrmse']=nrmse
    raw=HERE.parent/'data/raw_data.npz'
    values['density_observation_rms']=observation_rms(a['reconstruction'][0]**2,raw)
    values['emissivity_observation_rms']=observation_rms(a['emissivity'][0],raw)
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
