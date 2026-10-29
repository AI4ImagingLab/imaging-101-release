"""Independent standalone hydraulic-field evaluator; never imports solver code."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
HERE=Path(__file__).resolve().parent

def load_field(path):
    if not Path(path).is_file():raise ValueError('Missing NPZ: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if 'reconstruction' not in z:raise ValueError('Missing reconstruction')
        x=z['reconstruction']
    if x.shape!=(1,75,100) or x.dtype.kind!='f' or x.dtype.itemsize not in (4,8):raise ValueError('Expected real float32/float64(1,75,100)')
    if not np.isfinite(x).all():raise ValueError('Nonfinite field')
    return x.astype(np.float64)

def field_metrics(x,y):
    span=float(np.ptp(y));sy=float(abs(y).max());sx=float(abs(x).max())
    if span<=0 or sy==0 or not np.isfinite(span):raise ValueError('Degenerate hidden field')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        cosine=0. if sx==0 else float(np.vdot(x/sx,y/sy)/(np.linalg.norm(x/sx)*np.linalg.norm(y/sy)))
        error=float(np.sqrt(np.mean(((x-y)/span)**2)))
    if not np.isfinite(cosine) or not np.isfinite(error):raise ValueError('Metric overflow')
    return float(np.clip(cosine,-1,1)),error

def hydraulic_rms(field,raw_path):
    """Independent full30-experiment harmonic flux assembly, in physical units."""
    with np.load(raw_path,allow_pickle=False) as z:
        obs=z['hydraulic_head'];well=z['well_indices']
    if obs.shape!=(1,30,29) or obs.dtype.kind!='f' or not np.isfinite(obs).all():raise ValueError('Invalid evaluator observations')
    if well.shape!=(1,30,2) or well.dtype.kind not in 'iu' or len(np.unique(well[0],axis=0))!=30:raise ValueError('Invalid evaluator well calibration')
    well=well[0]
    if np.any(well[:,0]<0) or np.any(well[:,0]>=75) or np.any(well[:,1]<=0) or np.any(well[:,1]>=99):raise ValueError('Invalid well coordinates')
    with np.errstate(over='raise',invalid='raise',under='ignore'):
        k=np.exp(field[0])
        if not np.isfinite(k).all() or np.any(k<=0):raise ValueError('Conductivity outside representable positive domain')
        k=np.char.mod('%.6E',k.astype(np.float32)).astype(np.float32).astype(float)
    active=np.arange(7500).reshape(75,100)[:,1:99].ravel();lookup=np.full(7500,-1);lookup[active]=np.arange(7350)
    rows=[];cols=[];coefficients=[];rhs=np.full((7350,30),.1)
    for i,index in enumerate(active):
        r,c=divmod(int(index),100);diagonal=0.
        for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if rr<0 or rr>=75:continue
            weight=2*k[r,c]*k[rr,cc]/(k[r,c]+k[rr,cc]);diagonal+=weight
            if cc in [0,99]:rhs[i]+=weight*(120 if cc==0 else 110)
            else:rows.append(i);cols.append(int(lookup[rr*100+cc]));coefficients.append(-weight)
        rows.append(i);cols.append(i);coefficients.append(diagonal)
    ids=well[:,0]*100+well[:,1];rhs[lookup[ids],np.arange(30)]+=25.
    matrix=coo_matrix((coefficients,(rows,cols)),shape=(7350,7350)).tocsc();heads=spsolve(matrix,rhs).astype(np.float32)
    sampled=heads[lookup[ids]].T;selected=sampled[~np.eye(30,dtype=bool)].reshape(30,29)
    with np.errstate(over='raise',invalid='raise'):
        measured=float(np.sqrt(np.mean((selected.astype(float)-obs[0])**2)))
    if not np.isfinite(measured):raise ValueError('Nonfinite hydraulic observation metric')
    return measured

def evaluate(prediction,evaluation_dir=None):
    root=Path(evaluation_dir) if evaluation_dir is not None else HERE
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        config=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        for name in ['ncc_boundary','nrmse_boundary','forward_head_rms_boundary']:
            v=config[name]
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v):raise ValueError('Invalid numeric threshold '+name)
            if name=='ncc_boundary' and not -1<=v<=1:raise ValueError('Cosine threshold outside[-1,1]')
            if name!='ncc_boundary' and v<0:raise ValueError('Negative error threshold')
        reference=root/'ground_truth.npz'
        if not reference.exists():reference=root/'baseline_reference.npz'
        target=load_field(reference);estimate=load_field(prediction);cosine,error=field_metrics(estimate,target)
        forward=hydraulic_rms(estimate,root.parent/'data/raw_data.npz')
        values=dict(ncc=cosine,nrmse=error,forward_head_rms=forward)
        thresholds={key:dict(comparison='>=' if key=='ncc' else '<=',boundary=config[key+'_boundary']) for key in values}
        passed=all(values[k]>=v['boundary'] if v['comparison']=='>=' else values[k]<=v['boundary'] for k,v in thresholds.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=thresholds)
    except Exception as error:report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(error)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args();report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:print('Report write failure: '+str(error),file=sys.stderr);return 2
    return 0 if report['status']=='pass' else 1 if report['status']=='fail' else 2
if __name__=='__main__':raise SystemExit(main())
