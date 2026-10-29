"""Standalone evaluator: explicit declared arrays, no solver imports."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def relative_error(estimate,reference):
    norm=float(np.linalg.norm(reference.ravel()))
    if not np.isfinite(norm) or norm<=0:raise ValueError('Evaluator reference has zero or invalid norm')
    value=float(np.linalg.norm((estimate-reference).ravel())/norm)
    if not np.isfinite(value):raise ValueError('Nonfinite relative error')
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
def predicted_accelerance(arrays,frequency):
    natural=arrays['natural_frequency'][0];damping=arrays['damping_ratio'][0]
    if np.any(natural<10) or np.any(natural>5000) or np.any(np.diff(natural)<=0) or np.any(damping<=0) or np.any(damping>=1):raise ValueError('Require sorted in-band underdamped stable modes')
    poles=2*np.pi*natural*(-damping+1j*np.sqrt(1-damping*damping))
    w=2*np.pi*np.asarray(frequency)
    residues=arrays['reconstruction'][0]
    result=np.broadcast_to(arrays['lower_residual'][0,:,None],(6,len(w))).copy()-w[None,:]**2*arrays['upper_residual'][0,:,None]
    for i,pole in enumerate(poles):
        result-=w[None,:]**2*(residues[:,i,None]/(1j*w[None,:]-pole)+residues[:,i,None].conj()/(1j*w[None,:]-pole.conjugate()))
    return result


def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'));schema=config['output_schema']
        if set(schema)!={'natural_frequency','damping_ratio','reconstruction','fitted_accelerance','lower_residual','upper_residual'}:raise ValueError('Invalid output schema coverage')
        p=read_arrays(prediction,schema,'Prediction');reference=read_arrays(ROOT/'baseline_reference.npz',schema,'Attainable fit')
        with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as z:f=z['frequency'][0];observed=z['accelerance'][0]
        end=int(np.argmin(abs(f-5000)));start=int(np.argmin(abs(f-10)))
        output_grid=np.r_[np.arange(0,f[0],f[1]-f[0]),f[:end]]
        if len(output_grid)!=4999 or observed.shape!=(6,9999):raise ValueError('Invalid acquisition grid')
        scale=np.mean(abs(observed[:,start:end])**2,axis=1)
        if np.any(scale<=0):raise ValueError('Degenerate frequency-response observations')
        prediction_model=predicted_accelerance(p,f[start:end]);reference_model=predicted_accelerance(reference,f[start:end])
        pe=np.mean(abs(prediction_model-observed[:,start:end])**2,axis=1)/scale
        re=np.mean(abs(reference_model-observed[:,start:end])**2,axis=1)/scale
        reference_full=predicted_accelerance(reference,output_grid)
        if np.max(np.linalg.norm(reference['fitted_accelerance'][0]-reference_full,axis=1)/np.maximum(np.linalg.norm(reference_full,axis=1),1e-30))>1e-7:raise ValueError('Inconsistent attainable reference model')
        model=predicted_accelerance(p,output_grid)
        denominator=np.linalg.norm(model,axis=1)
        consistency=np.linalg.norm(p['fitted_accelerance'][0]-model,axis=1)/np.maximum(denominator,1e-30)
        measured={'maximum_response_mse_excess':float(np.max(pe-re)),'maximum_fitted_relative_l2':float(np.max(consistency))}
        bounds={key:config[key+'_boundary'] for key in measured}
        if not all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in bounds.values()):raise ValueError('Invalid numeric boundaries')
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
