import argparse,json,sys
from pathlib import Path
import numpy as np

def positive_number(value):
    return type(value) in (int,float) and np.isfinite(value) and value>0


def evaluate(prediction,root=None):
    root=Path(root) if root else Path(__file__).resolve().parent.parent
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        cfg=json.loads((root/'evaluation/metrics.json').read_text(encoding='utf-8'))
        raw_boundaries=cfg['objective_boundary']
        if type(raw_boundaries) is not list or len(raw_boundaries)!=2 or not all(positive_number(v) for v in raw_boundaries):raise ValueError('Invalid evaluator objective boundaries')
        boundaries=np.asarray(raw_boundaries,dtype=float)
        if boundaries.shape!=(2,) or not np.all(np.isfinite(boundaries)) or np.any(boundaries<=0):raise ValueError('Invalid evaluator objective boundaries')
        limit=cfg['forward_relative_l2_boundary']
        if not positive_number(limit):raise ValueError('Invalid evaluator consistency boundary')
        with np.load(root/'data/raw_data.npz',allow_pickle=False) as a:
            lag_us=a['lag_us'][0,1:];t=lag_us/1e6;g=a['correlation'][0,:,1:].T;r=a['radius_nm'][0]
        if t.shape!=(373,) or g.shape!=(373,2) or r.shape!=(200,) or not all(np.all(np.isfinite(v)) for v in [t,g,r]) or np.any(r<=0):raise ValueError('Invalid evaluator observations')
        with np.load(prediction,allow_pickle=False) as a:
            out={}
            for name,shape in [('reconstruction',(1,2,200)),('correlation',(1,2,373))]:
                if name not in a:raise ValueError('Missing output '+name)
                v=a[name]
                if v.shape!=shape:raise ValueError('Invalid shape '+name)
                if v.dtype!=np.dtype('float64'):raise ValueError('Expected real float64 '+name)
                if not np.all(np.isfinite(v)):raise ValueError('Nonfinite '+name)
                out[name]=v.copy()
        p=out['reconstruction'][0];reported=out['correlation'][0].T
        if np.any(p<0):raise ValueError('Negative radius bin mass')
        early=lag_us<5.0;beta=np.exp(np.polyfit(t[early],np.log(g[early]-1),2)[-1])
        if not np.all(np.isfinite(beta)) or np.any(beta<=0):raise ValueError('Invalid coherence estimate')
        q=4*np.pi*1.33/(817/1e9)*np.sin((150/180*np.pi)/2);s=6*np.pi*8.9e-4*(r/1e9)/(1.380649e-23*293*q*q);k=np.exp(-t[:,None]/s[None,:]);m=np.zeros((200,200))
        for j in range(1,199):m[j,j-1:j+2]=[-1,2,-1]
        expected=1+beta[None,:]*(k@p.T)**2
        with np.errstate(invalid='ignore'):field=np.sqrt((g-1)/beta)
        for j in range(2):
            bad=np.flatnonzero(g[:,j]<1.0);end=int(bad[0]) if len(bad) else len(t);v=p[j];res=k[:end]@v-field[:end,j]
            objective=float(res@res+1000*((v.sum()-1)**2+v[0]**2+v[-1]**2)+.1*np.sum((m@v)**2));consistency=float(np.linalg.norm(reported[:,j]-expected[:,j])/np.linalg.norm(expected[:,j]))
            report['metrics'][f'objective_{j}']=objective;report['metrics'][f'forward_relative_l2_{j}']=consistency
            report['thresholds'][f'objective_{j}']=dict(comparison='<=',boundary=float(boundaries[j]));report['thresholds'][f'forward_relative_l2_{j}']=dict(comparison='<=',boundary=float(limit))
        truth_path=root/'evaluation/ground_truth.npz'
        if truth_path.is_file():
            truth_limit=cfg.get('truth_relative_l2_boundary')
            if not positive_number(truth_limit):raise ValueError('Actual truth requires an explicit truth_relative_l2_boundary; method reference is not a fallback')
            with np.load(truth_path,allow_pickle=False) as truth:
                for name,shape in [('reconstruction',(1,2,200)),('correlation',(1,2,373))]:
                    ref=truth[name]
                    if ref.shape!=shape or ref.dtype!=np.dtype('float64') or not np.all(np.isfinite(ref)) or np.linalg.norm(ref)==0:raise ValueError('Invalid actual truth '+name)
                    key='truth_relative_l2_'+name;report['metrics'][key]=float(np.linalg.norm(out[name]-ref)/np.linalg.norm(ref));report['thresholds'][key]=dict(comparison='<=',boundary=float(truth_limit))
        if not all(np.isfinite(v) for v in report['metrics'].values()):raise ValueError('Nonfinite measured metric')
        report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in report['metrics'].items());report['status']='pass' if report['passed'] else 'fail';return report,0 if report['passed'] else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(exc)]);return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as e:print('Report write failed: '+str(e),file=sys.stderr);return 2
    return code
if __name__=='__main__':sys.exit(main())
