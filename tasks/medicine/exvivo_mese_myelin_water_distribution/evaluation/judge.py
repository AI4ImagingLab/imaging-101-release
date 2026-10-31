"""Standalone evaluator: independent direct complex-state RF algebra, no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,120,120,100),'myelin_fraction':(1,120,120),'total_density':(1,120,120),'predicted_echoes':(1,120,120,32)}
LAMBDA=0.003719667767540722

def number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
        raise ValueError('Invalid positive finite JSON number: '+name)
    return float(value)

def configuration():
    c=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
    return {k:number(c[k],k) for k in ['objective_sum_boundary','scaled_duality_gap_boundary','derived_consistency_boundary']}

def load_prediction(path,max_mass):
    with np.load(path,allow_pickle=False) as z:
        out={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing required key '+key)
            a=z[key]
            if a.shape!=shape or a.dtype!=np.dtype('float64'):raise ValueError('Expected float64 '+key+' shape '+str(shape))
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            if np.any(a<0):raise ValueError('Negative physical output '+key)
            out[key]=a[0]
    if np.max(out['reconstruction'])>max_mass:raise ValueError('Bin mass exceeds bound implied by public objective ceiling')
    if np.max(out['myelin_fraction'])>1:raise ValueError('Myelin fraction exceeds one')
    return out

def independent_dictionary(centroids):
    t2=np.logspace(np.log10(10.),np.log10(3000.),100);allA=[]
    for t1,kappa in centroids:
        state=np.zeros((100,3,257),complex);M=1-np.exp(-1000/t1);alpha=kappa*np.pi/2
        state[:,0,128]=-1j*np.sin(alpha)*M;state[:,1,128]=1j*np.sin(alpha)*M;state[:,2,128]=np.cos(alpha)*M
        e1=np.exp(-4.6/t1);e2=np.exp(-4.6/t2);c=np.cos(kappa*np.pi);s=np.sin(kappa*np.pi);a=(1+c)/2;b=(1-c)/2
        R=np.array([[a,-b,s],[-b,a,s],[-s/2,-s/2,c]])
        echoes=[]
        for _ in range(32):
            state[:,:2]*=e2[:,None,None];state[:,2]*=e1;state[:,2,128]+=1-e1
            state[:,0]=np.roll(state[:,0],2,axis=-1);state[:,1]=np.roll(state[:,1],-2,axis=-1)
            state=np.einsum('ij,bjk->bik',R,state)
            state[:,0]=np.roll(state[:,0],2,axis=-1);state[:,1]=np.roll(state[:,1],-2,axis=-1)
            state[:,:2]*=e2[:,None,None];state[:,2]*=e1;state[:,2,128]+=1-e1
            echoes.append(abs(state[:,0,128]))
        allA.append(np.array(echoes))
    return np.array(allA),t2

def measure(out,data,A,t2):
    mask=data['estimation_mask'][0].astype(bool);labels=data['cluster_labels'][0][mask].astype(int)-1
    y=data['echo_magnitudes'][0].mean(axis=-1)[mask];x=out['reconstruction'][mask]
    pred=np.empty_like(y);J=np.empty(len(y));gap=np.empty(len(y))
    for c,Ac in enumerate(A):
        m=labels==c;pred[m]=x[m]@Ac.T;u=pred[m]-y[m];g=u@Ac
        J[m]=np.sum(u*u,axis=1)+LAMBDA*np.sum(x[m]*x[m],axis=1)
        gap[m]=np.sum((np.sqrt(LAMBDA)*x[m]+np.minimum(g,0)/np.sqrt(LAMBDA))**2,axis=1)+2*np.sum(x[m]*np.maximum(g,0),axis=1)
    total=x.sum(axis=1);fraction=np.divide(x[:,(t2>15)&(t2<=40)].sum(axis=1),total,out=np.zeros_like(total),where=total>0)
    values={}
    for key,expected in [('total_density',total),('myelin_fraction',fraction),('predicted_echoes',pred)]:
        values[key+'_consistency']=float(np.max(abs(out[key][mask]-expected)/np.maximum(1.,abs(expected))))
    values['outside_mask_max']=max(float(np.max(abs(a[~mask]))) for a in out.values())
    values['objective_sum']=float(J.sum());values['scaled_duality_gap']=float(np.max(gap/np.maximum(J,1e-6)))
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined or nonfinite evaluation metric')
    return values

def evaluate(prediction):
    c=configuration();maxmass=np.sqrt(c['objective_sum_boundary']/LAMBDA)*(1+1e-10)
    out=load_prediction(prediction,maxmass)
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    A,t2=independent_dictionary(data['cluster_centroids'][0])
    # Actual truth, if legitimately supplied, takes precedence over method result.
    reference=HERE/'ground_truth.npz'
    if not reference.exists():reference=HERE/'baseline_reference.npz'
    ref=load_prediction(reference,maxmass);rm=measure(ref,data,A,t2)
    if rm['objective_sum']>c['objective_sum_boundary'] or rm['scaled_duality_gap']>c['scaled_duality_gap_boundary'] or max(rm[k] for k in rm if 'consistency' in k or k=='outside_mask_max')>c['derived_consistency_boundary']:
        raise ValueError('Evaluator reference fails declared physical/objective contract')
    m=measure(out,data,A,t2);thresholds={}
    for key in m:
        boundary=c['objective_sum_boundary'] if key=='objective_sum' else c['scaled_duality_gap_boundary'] if key=='scaled_duality_gap' else c['derived_consistency_boundary']
        thresholds[key]={'comparison':'<=','boundary':boundary}
    passed=all(m[k]<=v['boundary'] for k,v in thresholds.items())
    return dict(status='pass' if passed else 'fail',passed=bool(passed),metrics=m,thresholds=thresholds,errors=[]),0 if passed else 1

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args(argv)
    try:report,code=evaluate(Path(a.prediction))
    except Exception as exc:report=dict(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)]);code=2
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
