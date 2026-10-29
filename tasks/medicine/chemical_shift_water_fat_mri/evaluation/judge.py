"""Independent complete-array reference evaluation; never imports solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
SHAPE=(1,101,101,4)
DTYPES={'water':np.dtype('float32'),'fat':np.dtype('float32'),'fat_fraction':np.dtype('float32'),'fieldmap':np.dtype('float64'),'r2star':np.dtype('float64')}
def load_arrays(path):
    with np.load(path,allow_pickle=False) as archive:
        arrays={}
        for key,dtype in DTYPES.items():
            if key not in archive:raise ValueError('missing required key '+key)
            value=archive[key]
            if value.shape!=SHAPE or value.dtype!=dtype or not np.isfinite(value).all():raise ValueError(key+' must be finite '+str(dtype)+' '+str(SHAPE))
            arrays[key]=value
    return arrays
def numeric_metrics(estimate,reference):
    a=np.asarray(estimate,dtype=np.float64).ravel();b=np.asarray(reference,dtype=np.float64).ravel()
    scale=float(np.ptp(b));norm=float(np.linalg.norm(b))
    if scale<=0 or norm<=0:raise ValueError('reference requires positive norm and dynamic range')
    nrmse=float(np.sqrt(np.mean((a-b)**2))/scale)
    norm_a=float(np.linalg.norm(a));ncc=float(np.dot(a,b)/(norm_a*norm)) if norm_a else 0.
    return {'ncc':ncc,'nrmse':nrmse}
PERIOD=1/(.0032*1.494*42.576)


def signal_basis(rate):
    # Independent double arithmetic, source float32 spectral calibration retained.
    shifts=np.array([4.64655,5.3,4.31,2.76,2.1,1.3,.9],dtype=np.float32)
    weights=np.array([.048,.039,.004,.128,.693,.087],dtype=np.float32)
    omega=(2*np.pi*42.576*1.494*(shifts-shifts[0])).astype(np.float64)
    times=.00287+np.arange(3)*.0032
    fat=np.exp(1j*times[:,None]*omega[None,1:])@weights.astype(float)
    return np.column_stack((np.ones(3),fat))*np.exp(-np.arange(3)[:,None]*.0032*rate)


def projection_fit(Y,label,rate):
    # First echo phase belongs to the two complex coefficients.
    A=signal_basis(float(rate))*np.exp(2j*np.pi*np.arange(3)[:,None]*label/100)
    coefficients=np.linalg.lstsq(A,Y,rcond=None)[0]
    return coefficients,float(np.sum(abs(Y-A@coefficients)**2))


def graph_assets():
    with np.load(ROOT.parent/'data/raw_data.npz',allow_pickle=False) as z:Y=z['echoes'][0].reshape(3,-1).astype(np.complex128)
    if Y.shape!=(3,40804) or not np.isfinite(Y).all():raise ValueError('Invalid complex observations')
    J=np.empty((100,Y.shape[1]));A=signal_basis(40.)
    for label in range(100):
        design=A*np.exp(2j*np.pi*np.arange(3)[:,None]*label/100)
        residual=Y-design@np.linalg.lstsq(design,Y,rcond=None)[0]
        J[label]=np.sum(abs(residual)**2,axis=0)
    minima=np.argmin(J,axis=0);v=np.arange(Y.shape[1])
    curvature=np.maximum(0,J[(minima-1)%100,v]+J[(minima+1)%100,v]-2*J[minima,v])
    lattice=np.arange(Y.shape[1]).reshape(101,101,4);edges=[]
    for axis,distance in enumerate((1.5,1.5,5.)):
        lo=[slice(None)]*3;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        u=lattice[tuple(lo)].ravel();w=lattice[tuple(hi)].ravel()
        edges.append((u,w,.1*np.minimum(curvature[u],curvature[w])/distance))
    scale=float(np.sum(abs(Y)**2))
    if scale<=0:raise ValueError('Degenerate observed signal norm')
    return Y,J,edges,scale


def labels_rates(p):
    field=p['fieldmap'][0].ravel();rate=p['r2star'][0].ravel()
    labels=np.rint(field/PERIOD*100).astype(int)
    if np.any(field<0) or np.any(field>=PERIOD) or np.max(abs(field/PERIOD*100-labels))>1e-7 or np.any(labels>=100):raise ValueError('Field must be on the fixed 100-label alias grid')
    if np.any(rate<0) or np.any(rate>144) or np.max(abs(rate-np.rint(rate)))>1e-7:raise ValueError('Decay must be integer 0..144 per second')
    for k in ['water','fat','fat_fraction']:
        if np.any(p[k]<0):raise ValueError('Nonnegative amplitudes/fractions required')
    return labels,np.rint(rate).astype(int)


def graph_energy(labels,assets):
    Y,J,edges,scale=assets;total=float(np.sum(J[labels,np.arange(len(labels))]))
    for u,v,w in edges:
        delta=abs(labels[u]-labels[v]);delta=np.minimum(delta,100-delta)
        total+=float(np.sum(w*delta**2))
    return total/scale


def derived_metrics(p,assets):
    Y,J,edges,scale=assets;labels,rates=labels_rates(p)
    coefficients=np.empty((2,Y.shape[1]),complex);sse=0.
    for rate in np.unique(rates):
        mask=rates==rate
        # Batched independently demodulated least squares for equal decay rates.
        y=Y[:,mask]*np.exp(-2j*np.pi*np.arange(3)[:,None]*labels[None,mask]/100)
        A=signal_basis(rate);c=np.linalg.lstsq(A,y,rcond=None)[0]
        coefficients[:,mask]=c;sse+=float(np.sum(abs(y-A@c)**2))
    water=abs(coefficients[0]);fat=abs(coefficients[1])*float(np.sum(np.array([.048,.039,.004,.128,.693,.087],dtype=np.float32),dtype=np.float32))
    normalization=max(float(np.max(abs(Y))),np.finfo(float).tiny)
    consistency=max(float(np.max(abs(p['water'].ravel()-water))),float(np.max(abs(p['fat'].ravel()-fat))))/normalization
    fraction=100*p['fat'].astype(float)/(p['water'].astype(float)+p['fat'].astype(float)+np.finfo(float).eps)
    return {'profiled_echo_sse':sse/scale,'amplitude_consistency':consistency,'fraction_consistency':float(np.max(abs(p['fat_fraction']-fraction)))}


def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        c=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        names=['graph_energy_excess','profiled_echo_sse_excess','amplitude_consistency','fraction_consistency']
        for name in names:
            v=c[name+'_boundary']
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Invalid boundary '+name)
        if (ROOT/'ground_truth.npz').exists():raise ValueError('Unexpected actual truth requires truth-quality protocol certification')
        p=load_arrays(prediction);reference=load_arrays(ROOT/'baseline_reference.npz');assets=graph_assets()
        labels,_=labels_rates(p);ref_labels,_=labels_rates(reference)
        measured=derived_metrics(p,assets);attained=derived_metrics(reference,assets)
        if attained['amplitude_consistency']>c['amplitude_consistency_boundary'] or attained['fraction_consistency']>c['fraction_consistency_boundary']:raise ValueError('Physically inconsistent attainable reference')
        measured['profiled_echo_sse_excess']=measured.pop('profiled_echo_sse')-attained['profiled_echo_sse']
        measured['graph_energy_excess']=graph_energy(labels,assets)-graph_energy(ref_labels,assets)
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Nonfinite metric arithmetic')
        passed=all(measured[n]<=c[n+'_boundary'] for n in names)
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=measured,thresholds={n:dict(comparison='<=',boundary=c[n+'_boundary']) for n in names})
        return report,0 if passed else 1
    except Exception as error:report['errors']=[str(error)];report['metrics']={};return report,2


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as error:print('Cannot write report: '+str(error),file=sys.stderr);return 2
    return code
if __name__=='__main__':raise SystemExit(main())
