"""Standalone evaluator: independent preprocessing and physical-coordinate model."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,31),'predicted_360':(1,12,92),'predicted_197':(1,14,169)}
BOUNDS=np.array([[.01/1.5,.2/1.5],[.01/1.5,1/1.5],[1e-5,.18],[.0225,67.5]]+[[.001,10.]]*12+[[.0225,67.5]]+[[.001,10.]]*14)
THRESHOLDS=('weighted_sse','curve_360_relative','curve_197_relative')

def load_result(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(SHAPES):raise ValueError('Submission keys must exactly match the three required keys')
        out={k:f[k] for k in SHAPES}
    for key,shape in SHAPES.items():
        a=out[key]
        if a.shape!=shape or a.dtype.kind!='f' or a.dtype.itemsize!=8 or not np.isfinite(a).all():raise ValueError('Invalid shape, float64 dtype, or finite values for '+key)
    p=out['reconstruction'][0]
    if np.any(p<BOUNDS[:,0]) or np.any(p>BOUNDS[:,1]):raise ValueError('Parameters outside declared physical support')
    return out

def observations(path):
    with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    expected=set();out={}
    for n,qlow,qhigh,divisor,scale,nedge,ne in [('360',.6,1.8,8.,.5,31,92),('197',.55,1.58,9.,1.,30,169)]:
        specs={'signal':((1,3,249,320),'float64'),'standard_error':((1,3,249,320),'float64'),'q_edges':((1,250),'float64'),'energy_edges':((1,321),'float64'),'analysis_q_edges':((1,nedge),'float64'),'analysis_energy_index':((1,ne),'int64'),'acquisition_mask':((1,3,249,320),'bool')}
        for key,(shape,dtype) in specs.items():
            key=key+'_'+n;expected.add(key);a=raw[key]
            if a.shape!=shape or a.dtype!=np.dtype(dtype):raise ValueError('Invalid calibration format '+key)
            if not key.startswith(('signal_','standard_error_')) and not np.isfinite(a).all():raise ValueError('Nonfinite calibration')
        signal=raw['signal_'+n][0];error=raw['standard_error_'+n][0];mask=raw['acquisition_mask_'+n][0]
        if not np.array_equal(np.isfinite(signal),mask) or not np.array_equal(np.isfinite(error),mask) or np.isinf(signal).any() or np.isinf(error).any() or np.any(error[mask]<0):raise ValueError('Invalid acquisition missingness mask')
        old=raw['q_edges_'+n][0];new=raw['analysis_q_edges_'+n][0]
        if np.any(np.diff(old)<=0) or np.any(np.diff(new)<=0):raise ValueError('Nonincreasing Q edges')
        centers=(new[1:]+new[:-1])/2;select=np.flatnonzero((centers>=qlow)&(centers<=qhigh))
        oldmid=(old[:-1]+old[1:])/2;ei=raw['analysis_energy_index_'+n][0]
        if not np.array_equal(ei,np.arange(114,206) if n=='360' else np.arange(76,245)):raise ValueError('Changed fixed acquisition window')
        values=[];variance=[]
        for j in select:
            overlap=np.maximum(0,np.minimum(new[j+1],old[1:])-np.maximum(new[j],old[:-1]))/np.diff(old)
            keep=(overlap>0)&(oldmid>=qlow)&(oldmid<=qhigh);w=overlap[keep]
            values.append(np.sum(signal[:,keep,:][:,:,ei]*w[None,:,None],axis=1)/divisor)
            variance.append(np.sum(error[:,keep,:][:,:,ei]**2*w[None,:,None]**2,axis=1)/divisor**2)
        v=np.stack(values,axis=1);var=np.stack(variance,axis=1)
        y=(v[0]-v[2])*scale;sd=np.sqrt(var[0]+var[2])*scale;r=v[1]-v[2]
        e=raw['energy_edges_'+n][0];energy=((e[1:]+e[:-1])/2)[ei]
        if not all(np.isfinite(a).all() for a in [y,sd,r,energy]) or np.any(sd<=0) or np.any(r.sum(axis=1)<=0):raise ValueError('Invalid finite analysis cells')
        out[n]=(centers[select],energy,y,sd,r/r.sum(axis=1)[:,None])
    if set(raw)!=expected:raise ValueError('Unexpected public data keys')
    return out

def physical_forward(p,data):
    predictions={}
    for n,index,start,stop in [('360',3,4,16),('197',16,17,31)]:
        q,e,y,sd,r=data[n];step=e[1]-e[0];omega=np.arange(1-e.size,e.size)*step
        x=q*2.48
        j0=np.sin(x)/x;j1=np.sin(x)/x**2-np.cos(x)/x;j2=(3/x**3-1/x)*np.sin(x)-3*np.cos(x)/x**2
        def L(g):return g/(np.pi*(omega**2+g**2))
        rows=[]
        for i,Q in enumerate(q):
            t=np.exp(-Q**2*p[2]/3)*L(p[0]*Q**2)
            rot=3*j1[i]**2*L(p[1])+5*j2[i]**2*L(3*p[1]);rot[e.size-1]+=j0[i]**2/step
            molecular=np.convolve(rot,t,mode='same')
            rows.append(np.convolve(r[i],molecular/p[index]+p[start+i],mode='valid'))
        predictions[n]=np.asarray(rows)
    return predictions

def evaluate(prediction,evaluation_dir=HERE):
    evaluation_dir=Path(evaluation_dir)
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((evaluation_dir/'metrics.json').read_text(encoding='utf8'))
        if not isinstance(config,dict) or {k for k in config if k.endswith('_boundary')}!={k+'_boundary' for k in THRESHOLDS}:raise ValueError('Boundary keys must exactly match the three declared metrics')
        if 'comparisons' in config and config['comparisons']!={k:'<=' for k in THRESHOLDS}:raise ValueError('Comparisons must contain exactly all three declared upper bounds')
        for key in THRESHOLDS:
            value=config[key+'_boundary']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:raise ValueError('Invalid positive finite JSON-number boundary '+key)
            report['thresholds'][key]={'comparison':'<=','boundary':value}
        reference=evaluation_dir/'ground_truth.npz'
        if not reference.exists():reference=evaluation_dir/'baseline_reference.npz'
        ref=load_result(reference)
        submitted=load_result(prediction)
        data=observations(evaluation_dir.parent/'data/raw_data.npz')
        expected=physical_forward(submitted['reconstruction'][0],data)
        measured={}
        measured['weighted_sse']=float(sum(np.sum(((submitted['predicted_'+n][0]-data[n][2])/data[n][3])**2) for n in data))
        for n in data:
            norm=float(np.linalg.norm(expected[n]))
            if norm<=0 or not np.isfinite(norm):raise ValueError('Degenerate public forward norm')
            measured['curve_'+n+'_relative']=float(np.linalg.norm(submitted['predicted_'+n][0]-expected[n])/norm)
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite computed metric')
        report['metrics']=measured
        passed=all(measured[k]<=report['thresholds'][k]['boundary'] for k in THRESHOLDS)
        report.update(status='pass' if passed else 'fail',passed=bool(passed))
        return report,0 if passed else 1
    except Exception as exc:
        report['metrics']={};report['errors']=[str(exc)];return report,2

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
