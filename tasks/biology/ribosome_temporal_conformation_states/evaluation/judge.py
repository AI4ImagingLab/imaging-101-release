"""Independent Gaussian-HMM evaluator with joint state canonicalization."""
from pathlib import Path
import sys,argparse,json
import numpy as np
from scipy.special import logsumexp
HERE=Path(__file__).resolve().parent
SHAPES={'reconstruction':(1,590,500),'state_mean':(1,2),'state_variance':(1,2),'transition_probability':(1,2,2),'initial_probability':(1,2),'population_fraction':(1,2)}
def load_arrays(path,starts,stops):
    if not Path(path).is_file():raise ValueError('File not found: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for key,shape in SHAPES.items():
            if key not in z:raise ValueError('Missing '+key)
            a=z[key]
            if a.shape!=shape:raise ValueError('Wrong shape '+key)
            dtype='int64' if key=='reconstruction' else 'float64'
            if a.dtype!=np.dtype(dtype):raise ValueError('Expected '+dtype+' for '+key)
            if not np.isfinite(a).all():raise ValueError('Nonfinite '+key)
            arrays[key]=a[0].copy()
    valid=(np.arange(500)[None]>=starts[:,None])&(np.arange(500)[None]<stops[:,None])
    path=arrays['reconstruction']
    if np.any(path[~valid]!=-1) or np.any((path[valid]<0)|(path[valid]>1)):raise ValueError('Invalid state/sentinel or native-frame interval')
    if np.any(arrays['state_variance']<=0):raise ValueError('Variance must be positive')
    for key in ['transition_probability','initial_probability','population_fraction']:
        a=arrays[key]
        if np.any((a<0)|(a>1)):raise ValueError('Invalid probability '+key)
        if not np.allclose(a.sum(axis=-1),1,rtol=0,atol=1e-8):raise ValueError('Probability normalization '+key)
    # A global label permutation changes no HMM physics. Sort estimated means
    # without reference fitting, and apply the SAME permutation to all quantities.
    order=np.argsort(arrays['state_mean']);old_to_new=np.empty(2,dtype=np.int64);old_to_new[order]=np.arange(2)
    for key in ['state_mean','state_variance','initial_probability','population_fraction']:arrays[key]=arrays[key][order]
    arrays['transition_probability']=arrays['transition_probability'][order][:,order]
    path[valid]=old_to_new[path[valid]]
    return arrays,valid
def likelihood_and_decode(sequences,model):
    mu=model['state_mean'];var=model['state_variance'];A=model['transition_probability'];pi=model['initial_probability'];total=0.;paths=[]
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        logA=np.log(A);logpi=np.log(pi)
        for x in sequences:
            emission=-.5*np.log(2*np.pi*var)-.5*(x[:,None]-mu)**2/var
            alpha=logpi+emission[0];delta=alpha.copy();pointers=np.zeros((len(x),2),dtype=np.int64)
            for t in range(1,len(x)):
                alpha=emission[t]+logsumexp(alpha[:,None]+logA,axis=0)
                pair=delta[:,None]+logA;pointers[t]=np.argmax(pair,axis=0);delta=emission[t]+np.max(pair,axis=0)
            total+=float(logsumexp(alpha));path=np.empty(len(x),dtype=np.int64);path[-1]=int(np.argmax(delta))
            for t in range(len(path)-2,-1,-1):path[t]=pointers[t+1,path[t+1]]
            paths.append(path)
    return float(total),np.concatenate(paths)
def posterior_population(sequences,model):
    mu=model['state_mean'];var=model['state_variance'];logA=np.log(model['transition_probability']);logpi=np.log(model['initial_probability']);counts=np.zeros(2);total=0
    for observations in sequences:
        emission=-.5*np.log(2*np.pi*var)-.5*(observations[:,None]-mu)**2/var
        alpha=np.empty_like(emission);alpha[0]=logpi+emission[0]
        for t in range(1,len(observations)):alpha[t]=emission[t]+logsumexp(alpha[t-1,:,None]+logA,axis=0)
        logz=logsumexp(alpha[-1]);beta=np.zeros_like(alpha)
        for t in range(len(observations)-2,-1,-1):beta[t]=logsumexp(logA+emission[t+1][None]+beta[t+1][None],axis=1)
        counts+=np.exp(alpha+beta-logz).sum(0);total+=len(observations)
    return counts/total


def log_parameter_prior(model,observations):
    from scipy.special import gammaln
    centres=np.quantile(observations,[1/3,2/3]);precision=1/model['state_variance'];mu=model['state_mean']
    gamma=2.5*np.log(.01)-gammaln(2.5)+1.5*np.log(precision)-.01*precision
    normal=.5*np.log(.25*precision/(2*np.pi))-.125*precision*(mu-centres)**2
    return float(np.sum(gamma+normal))


def evaluate(prediction,evaluation_dir=None):
    directory=HERE if evaluation_dir is None else Path(evaluation_dir)
    result=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        rules={'mean_logposterior_deficit':'<=','observation_loglikelihood_per_frame':'>=','decoder_agreement':'>='}
        thresholds={k:dict(comparison=v,boundary=cfg[k+'_boundary']) for k,v in rules.items()}
        if not all(isinstance(v['boundary'],(int,float)) and np.isfinite(v['boundary']) for v in thresholds.values()):raise ValueError('Invalid evaluator boundary')
        with np.load(directory.parent/'data/raw_data.npz',allow_pickle=False) as z:raw=z['intensity'][0];lo=z['interval_start'][0];hi=z['interval_stop'][0]
        if raw.shape!=(590,500,2) or lo.shape!=(590,) or hi.shape!=(590,) or not np.isfinite(raw).all():raise ValueError('Malformed acquisition assets')
        if np.any(lo<0)|np.any(hi>500)|np.any(hi<=lo):raise ValueError('Invalid native intervals')
        refpath=directory/'ground_truth.npz'
        if not refpath.exists():refpath=directory/'baseline_reference.npz'
        ref,valid=load_arrays(refpath,lo,hi);pred,_=load_arrays(prediction,lo,hi)
        m={}
        sequences=[]
        for trace,start,stop in zip(raw,lo,hi):
            row=trace[start:stop];total=row.sum(1)
            if np.any(total==0):raise ValueError('Invalid acquisition intensity denominator')
            sequences.append(row[:,1]/total)
        loglikelihood,decoded=likelihood_and_decode(sequences,pred);submitted=pred['reconstruction'][valid]
        m['observation_loglikelihood_per_frame']=float(loglikelihood/len(submitted));m['decoder_agreement']=float(np.mean(submitted==decoded))
        observed=np.concatenate(sequences);reference_loglikelihood,_=likelihood_and_decode(sequences,ref)
        candidate=loglikelihood+log_parameter_prior(pred,observed)
        standard=reference_loglikelihood+log_parameter_prior(ref,observed)
        m['mean_logposterior_deficit']=float(max(0.,standard-candidate)/len(observed))
        population=posterior_population(sequences,pred);standard_population=posterior_population(sequences,ref)
        m['population_posterior_discrepancy']=float(np.max(abs(pred['population_fraction']-population)))
        attained=float(np.max(abs(ref['population_fraction']-standard_population)))
        thresholds['population_posterior_discrepancy']=dict(comparison='<=',boundary=max(1e-6,1.1*attained))

        if not all(np.isfinite(v) for v in m.values()):raise ValueError('Numerically undefined HMM metric')
        passed=all(m[k]>=v['boundary'] if v['comparison']=='>=' else m[k]<=v['boundary'] for k,v in thresholds.items())
        result.update(status='pass' if passed else 'fail',passed=passed,metrics=m,thresholds=thresholds);return result,0 if passed else 1
    except Exception as exc:result['errors']=[str(exc)];return result,2
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report-write failure: '+str(exc),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
