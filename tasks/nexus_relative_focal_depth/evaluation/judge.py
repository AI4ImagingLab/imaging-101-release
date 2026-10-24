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


def evaluate(prediction):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        config=json.loads((HERE/'metrics.json').read_text(encoding='utf8'))
        actual=HERE/'ground_truth.npz'
        reference=load_field(actual if actual.exists() else HERE/'baseline_reference.npz','Evaluator')
        metrics(reference,reference)
        thresholds={}
        for key,direction in [('ncc','greater_equal'),('nrmse','less_equal')]:
            value=config[key+'_boundary']
            if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('Invalid metric boundary')
            if key=='nrmse' and value<=0:raise ValueError('Reference agreement requires nonzero numerical tolerance')
            if key=='ncc' and not(-1<=value<=1):raise ValueError('NCC boundary outside[-1,1]')
            thresholds[key]={'comparison':direction,'boundary':value}
    except Exception as error:
        report['errors']=[f'Evaluator error: {error}'];return report,2
    try:values=metrics(load_field(prediction,'Prediction'),reference)
    except Exception as error:
        report['errors']=[str(error)];return report,2
    passed=values['ncc']>=thresholds['ncc']['boundary'] and values['nrmse']<=thresholds['nrmse']['boundary']
    report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
    return report,0 if passed else 1


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
