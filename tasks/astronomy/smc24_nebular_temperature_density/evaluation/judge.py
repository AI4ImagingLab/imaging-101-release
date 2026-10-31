"""Standalone four-diagnostic plasma-condition evaluator."""
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def validate(path):
    with np.load(path,allow_pickle=False) as a:
        if 'reconstruction' not in a:raise ValueError('Missing reconstruction key')
        x=a['reconstruction']
        if x.shape!=(1,4,2) or x.dtype!=np.float64:raise ValueError('Expected real float64 (1,4,2)')
        if not np.all(np.isfinite(x)):raise ValueError('Nonfinite conditions')
    return x

def physical_ratios(x):
    """Independent finite-level balance evaluation from public atomic calibration."""
    with np.load(HERE.parent/'data/raw_data.npz',allow_pickle=False) as z:
        raw={k:z[k][0] for k in z.files}
    def attenuation(wave):
        u=1e4/wave
        if u<=.3:return 0.
        if u<=1.1:return 3.1*.574*u**1.61-.527*u**1.61
        v=u-1.82
        a=np.polynomial.polynomial.polyval(v,[1,.17699,-.50447,-.02427,.72085,.01979,-.7753,.32999])
        b=np.polynomial.polynomial.polyval(v,[0,1.41338,2.28305,1.07233,-5.38434,-.622510,5.3026,-2.09002])
        return 3.1*a+b
    definitions=[('N2',(4,3),(3,2),'N2_5755A','N2_6584A',5755.,6584.),('O2',(2,0),(1,0),'O2_3726A','O2_3729A',3726.,3729.),('O3',(4,3),(3,2),'O3_4363A','O3_5007A',4363.,5007.),('S2',(1,0),(2,0),'S2_6731A','S2_6716A',6731.,6716.)]
    def ratio(definition,T,n):
        ion,up,down,f1,f2,w1,w2=definition
        energy=raw[ion+'_energy'];g=raw[ion+'_weight'];N=len(g);rates=raw[ion+'_A'].copy();strength=raw[ion+'_collision_strength'];nodes=raw[ion+'_collision_temperature']
        if not 10**nodes[0]<=T<=10**nodes[-1] or not 1<=n<=1e8:return None
        for lo in range(N-1):
            for hi in range(lo+1,N):
                omega=np.interp(np.log10(T),nodes,strength[lo,hi]);q=8.629132419846745e-6*omega/(g[hi]*np.sqrt(T))
                rates[hi,lo]+=n*q
                rates[lo,hi]+=n*q*g[hi]/g[lo]*np.exp((energy[lo]-energy[hi])/(6.95034760642295e-9*T))
        generator=rates.T-np.diag(rates.sum(1));generator[-1,:]=1.;rhs=np.zeros(N);rhs[-1]=1.;pop=np.linalg.solve(generator,rhs)
        def line(pair):
            u,l=pair;return pop[u]*raw[ion+'_A'][u,l]*(energy[u]-energy[l])
        predicted=line(up)/line(down)
        observed=float(raw['flux_'+f1][0]/raw['flux_'+f2][0])*10**(.4*.0325583778321743011*(attenuation(w1)-attenuation(w2)))
        return float(predicted/observed-1)
    errors=[]
    for k,(td,nd) in enumerate([(0,1),(0,3),(2,1),(2,3)]):
        T,n=x[0,k];v=[ratio(definitions[j],T,n) for j in (td,nd)]
        errors.append([1e100 if q is None else abs(q) for q in v])
    return np.asarray(errors)

def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        x=validate(path);spec=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        boundary=spec['max_fractional_line_ratio_residual_boundary']
        if isinstance(boundary,bool) or not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid boundary')
        errors=physical_ratios(x)
        values={'max_fractional_line_ratio_residual':float(errors.max())}
        if not np.isfinite(errors).all():raise ValueError('Nonfinite atomic model evaluation')
        passed=values['max_fractional_line_ratio_residual']<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds={'max_fractional_line_ratio_residual':{'comparison':'<=','boundary':boundary}})
        return report,0 if passed else 1
    except Exception as exc:report['errors']=[str(exc)];return report,2


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:print('Report write failed: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
