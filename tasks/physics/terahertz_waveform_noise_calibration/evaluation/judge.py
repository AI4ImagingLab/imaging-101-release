"""Standalone evaluator: explicit declared arrays, no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def relative_error(estimate,reference):
    norm=float(np.linalg.norm(reference.ravel()))
    if not np.isfinite(norm) or norm<=0:raise ValueError('Evaluator reference has zero or invalid norm')
    with np.errstate(over='ignore',invalid='ignore'):
        value=float(np.linalg.norm((estimate-reference).ravel())/norm)
    if not np.isfinite(value):raise ValueError('Prediction relative error overflowed or is nonfinite')
    return value
def read_arrays(path,schema,label):
    if not Path(path).is_file():raise ValueError(f'{label} file does not exist: {path}')
    with np.load(path,allow_pickle=False) as f:
        result={}
        for key,spec in schema.items():
            if key not in f:raise ValueError(f'{label} missing required key {key}')
            v=f[key]
            if list(v.shape)!=spec['shape']:raise ValueError(f'{label} {key} shape {v.shape}, expected {spec["shape"]}')
            if str(v.dtype)!=spec['dtype']:raise ValueError(f'{label} {key} dtype {v.dtype}, expected {spec["dtype"]}')
            if not np.isfinite(v).all():raise ValueError(f'{label} {key} contains NaN or infinity')
            result[key]=v
    return result
def observable_quantities(p):
    mu=p['reconstruction'][0];amplitude=p['amplitude'][0];delay=p['delay'][0];noise=p['noise'][0]
    if amplitude[0]!=1 or delay[0]!=0 or np.any(amplitude<=0) or np.any(noise<=0):raise ValueError('Invalid drift gauge or positive amplitudes/noise')
    frequency=2*np.pi*np.fft.rfftfreq(267,d=.05003462406015038)
    spectrum=np.fft.rfft(mu)[None,:]*np.exp(-1j*delay[:,None]*frequency[None,:])*amplitude[:,None]
    expected=np.fft.irfft(spectrum,n=267,axis=1)
    derivative=np.fft.irfft(1j*frequency[None,:]*spectrum,n=267,axis=1)
    variance=noise[0]**2+(noise[1]*expected)**2+(noise[2]*derivative)**2
    return expected,np.sqrt(variance),variance*(49/50)


def likelihood(p,observed):
    expected,std,variance=observable_quantities(p)
    if not np.isfinite(variance).all() or np.any(variance<=0):raise ValueError('Invalid physical variance')
    return float(.5*np.mean(np.log(variance)+(observed-expected)**2/variance))


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'));schema=config['output_schema']
        if set(schema)!={'reconstruction','amplitude','delay','noise','expected_waveforms','noise_std'}:raise ValueError('Invalid output schema coverage')
        p=read_arrays(prediction,schema,'Prediction');reference=read_arrays(ROOT/'baseline_reference.npz',schema,'Attainable fit')
        with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as z:observed=z['waveforms']
        if observed.shape!=(1,50,267) or not np.isfinite(observed).all():raise ValueError('Invalid repeated observations')
        expected,std,variance=observable_quantities(p)
        measured={'mean_nll_excess':likelihood(p,observed[0])-likelihood(reference,observed[0]),'expected_relative_l2':float(np.linalg.norm(p['expected_waveforms'][0]-expected)/max(np.linalg.norm(expected),1.)),'noise_std_relative_l2':float(np.linalg.norm(p['noise_std'][0]-std)/np.linalg.norm(std))}
        bounds={key:config[key+'_boundary'] for key in measured}
        if not all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in bounds.values()):raise ValueError('Invalid numeric boundary')
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite metric')
        passed=all(measured[key]<=value for key,value in bounds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds={k:dict(comparison='<=',boundary=v) for k,v in bounds.items()})
    except Exception as error:report['errors']=[str(error)]
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':sys.exit(main())
