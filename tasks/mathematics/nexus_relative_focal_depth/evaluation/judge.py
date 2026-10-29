"""Independent strict NPZ evaluator; never imports a submitted solver."""
from pathlib import Path
import argparse,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SHAPE=(1,1080,1920)


def load_field(path,role):
    if not Path(path).is_file():raise ValueError(f'{role}: missing file {path}')
    with np.load(path,allow_pickle=False) as file:
        if 'reconstruction' not in file.files:raise ValueError(f'{role}: missing key reconstruction')
        field=file['reconstruction']
    if field.shape!=SHAPE:raise ValueError(f'{role}: expected shape {SHAPE}, received {field.shape}')
    if field.dtype.kind!='f' or field.dtype.itemsize not in (4,8):raise ValueError(f'{role}: expected real float32 or float64')
    if not np.isfinite(field).all():raise ValueError(f'{role}: nonfinite values')
    if np.any((field<-10)|(field>10)):raise ValueError(f'{role}: relative focal coordinate outside[-10,10]')
    return field.astype(np.float64)


def metrics(estimate,reference):
    a=estimate.ravel();b=reference.ravel();norm=np.linalg.norm(b);span=np.ptp(b)
    if not(norm>0 and span>0):raise ValueError('Degenerate evaluator reference')
    scale=np.linalg.norm(a)
    values={'ncc':float(np.dot(a,b)/(scale*norm)) if scale>0 else 0.,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}
    if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined metric')
    return values


def contrast_polynomials(raw_path):
    """Observation-defined contrast polynomial, stable float64 least squares."""
    with np.load(raw_path,allow_pickle=False) as z:frames=z['observations']
    if frames.shape!=(1,19,1080,1920,3) or frames.dtype!=np.uint8:raise ValueError('Invalid public focal stack')
    sharp=np.empty((19,1080,1920),dtype=float)
    for k,frame in enumerate(frames[0]):
        image=frame.astype(float)/255.;pad=np.pad(image,((1,1),(1,1),(0,0)),mode='edge')
        sharp[k]=(np.abs(pad[1:-1,:-2]-2*image+pad[1:-1,2:])+np.abs(pad[:-2,1:-1]-2*image+pad[2:,1:-1])).sum(axis=-1)
    normalized=sharp/(sharp.max(axis=0)+.3)
    nodes=np.linspace(-1,1,19);basis=np.vander(nodes,9)
    coefficients=(np.linalg.pinv(basis)@normalized.reshape(19,-1)).reshape(9,1080,1920)
    return coefficients


def field_energy(field,coefficients):
    u=field[0];value=np.zeros_like(u)
    for c in coefficients:value=value*(u/10)+c
    gx=np.zeros_like(u);gy=np.zeros_like(u);gx[:,:-1]=np.diff(u,axis=1);gy[:-1]=np.diff(u,axis=0)
    return float(np.mean(-6*value+np.hypot(gx,gy)))


def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        boundary=config['mean_energy_excess_boundary']
        if type(boundary) not in (int,float) or not np.isfinite(boundary) or boundary<0:raise ValueError('Invalid energy boundary')
        p=load_field(prediction,'Prediction');reference=load_field(HERE/'baseline_reference.npz','Attainable fit')
        raw=HERE.parent/'data/raw_data.npz'
        coefficients=contrast_polynomials(raw)
        pe=field_energy(p,coefficients);re=field_energy(reference,coefficients)
        values=dict(mean_energy=pe,attainable_mean_energy=re,mean_energy_excess=pe-re)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite energy')
        passed=pe-re<=boundary
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds={'mean_energy_excess':dict(comparison='<=',boundary=boundary)})
        return report,0 if passed else 1
    except Exception as error:
        report['errors']=[str(error)];return report,2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction',required=True,type=Path);parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args();report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:
        print(f'Cannot write report: {error}',file=sys.stderr);return 2
    return code


if __name__=='__main__':sys.exit(main())
