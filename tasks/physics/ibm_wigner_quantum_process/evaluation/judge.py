"""Standalone complex process evaluator. Never imports a solver."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def score(prediction, reference):
    """Global-phase aligned complex-entry RMS and Frobenius unitarity error."""
    inner=np.vdot(reference,prediction)
    aligned=prediction*np.exp(-1j*np.angle(inner))
    return {'phase_aligned_process_rmse':float(np.sqrt(np.mean(np.abs(aligned-reference)**2))),
            'unitarity_error':float(np.linalg.norm(prediction[0].conj().T@prediction[0]-np.eye(2)))}

def checked(path):
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z: raise ValueError('Missing reconstruction key')
        x=z['reconstruction']
    if x.shape!=(1,2,2): raise ValueError('Expected shape (1,2,2)')
    if x.dtype.kind!='c' or x.dtype.itemsize not in (8,16): raise ValueError('Expected complex64 or complex128')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite reconstruction')
    return x

def observation_loss(matrix,raw):
    """Ideal-model sphere-average complex SSE; global process phase cancels."""
    u=matrix[0]*np.exp(-.5j*np.angle(np.linalg.det(matrix[0])))
    q=np.array([(u[0,1].imag+u[1,0].imag)/2,(u[0,1].real-u[1,0].real)/2,(u[0,0].imag-u[1,1].imag)/2,(u[0,0].real+u[1,1].real)/2])
    norm=np.linalg.norm(q)
    if norm<=0:raise ValueError('Degenerate process matrix')
    q/=norm;theta=raw['polar_angle'][0];phi=raw['azimuth'][0];w=raw['quadrature_weight'][0]
    if theta.shape!=(50,) or phi.shape!=(50,) or w.shape!=(50,) or np.any(w<=0) or abs(w.sum()-1)>1e-10:raise ValueError('Invalid sphere geometry or weights')
    xyz=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    v=q[:3]@xyz;d=q[3];e=np.empty((2,4,50),complex)
    e[0,0]=d*d;e[1,0]=1j*d*v;e[0,1:]=-1j*q[:3,None]*d;e[1,1:]=q[:3,None]*v
    observed=raw['expectation'][0]
    if observed.shape!=(2,4,50) or not np.isfinite(observed).all():raise ValueError('Invalid measured expectations')
    return float(np.sum(abs(e-observed)**2*w[None,None,:])/8)


def evaluate(path):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        root=Path(__file__).resolve().parent
        config=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        limits={k:{'comparison':'<=','boundary':config[k+'_boundary']} for k in ['complex_mse_excess','unitarity_error']}
        if not all(type(v['boundary']) in (int,float) and np.isfinite(v['boundary']) and v['boundary']>0 for v in limits.values()): raise ValueError('Invalid evaluator boundaries')
        truth=root/'ground_truth.npz'
        reference=checked(truth if truth.exists() else root/'baseline_reference.npz')
        prediction=checked(path)
        with np.load(root.parent/'data/raw_data.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        metrics={'complex_mse_excess':observation_loss(prediction,raw)-observation_loss(reference,raw),'unitarity_error':float(np.linalg.norm(prediction[0].conj().T@prediction[0]-np.eye(2)))}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite metric')
        passed=all(metrics[k]<=v['boundary'] for k,v in limits.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=metrics,thresholds=limits)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)]
        return report,2

def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report,code=evaluate(a.prediction)
    try:
        path=Path(a.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
