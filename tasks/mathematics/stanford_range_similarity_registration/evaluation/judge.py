"""Independent strict submission evaluator; never imports solver code."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ASSETS=Path(__file__).resolve().parent

def relative_l2(estimate,reference):
    norm=np.linalg.norm(reference.ravel())
    if norm<=0 or not np.isfinite(norm):raise ValueError('Undefined evaluator reference norm')
    return float(np.linalg.norm((estimate-reference).ravel())/norm)

def evaluate(prediction,asset_dir=ASSETS,public_path=None):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        assets=Path(asset_dir);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
        specs=config['outputs']
        if not specs:raise ValueError('Empty output configuration')
        target=assets/'ground_truth.npz'
        if not target.exists():target=assets/'baseline_reference.npz'
        with np.load(target,allow_pickle=False) as f:ref={k:f[k] for k in specs}
        with np.load(prediction,allow_pickle=False) as f:pred={k:f[k] for k in specs}
        for label,arrays in [('prediction',pred),('evaluator reference',ref)]:
            for k,spec in specs.items():
                a=arrays[k]
                if list(a.shape)!=spec['shape'] or a.dtype!=np.dtype(spec['dtype']):raise ValueError('Invalid '+label+' shape/dtype: '+k)
                if not np.isfinite(a).all():raise ValueError('Nonfinite '+label+': '+k)
        measured={};limits={}
        for k,spec in specs.items():
            boundary=spec['relative_l2_boundary']
            if not np.isfinite(boundary) or boundary<=0:raise ValueError('Invalid metric boundary')
            measured[k+'_relative_l2']=relative_l2(pred[k],ref[k]);limits[k+'_relative_l2']={'comparison':'<=','boundary':boundary}
        public=Path(public_path) if public_path is not None else ASSETS.parent/'data/raw_data.npz'
        with np.load(public,allow_pickle=False) as f:x=f['target_points'][0];y=f['source_points'][0]
        r=pred['rotation'][0];s=pred['scale'].item();t=pred['translation'][0];aligned=pred['reconstruction'][0]
        if s<=0:raise ValueError('Similarity scale must be positive')
        measured['transform_consistency_m']=float(np.sqrt(np.mean((aligned-(s*(y@r)+t))**2)))
        measured['orthogonality_frobenius']=float(np.linalg.norm(r.T@r-np.eye(3)))
        measured['rotation_determinant_error']=float(abs(np.linalg.det(r)-1))
        distances=cKDTree(x).query(aligned,workers=1)[0]
        measured['nearest_rms_m']=float(np.sqrt(np.mean(distances**2)))
        for k in ['transform_consistency_m','orthogonality_frobenius','rotation_determinant_error','nearest_rms_m']:
            value=config[k+'_boundary']
            if not np.isfinite(value) or value<=0:raise ValueError('Invalid physical boundary')
            limits[k]={'comparison':'<=','boundary':value}
        if not all(np.isfinite(v) for v in measured.values()):raise ValueError('Undefined metric')
        passed=all(v<=limits[k]['boundary'] for k,v in measured.items())
        report.update(status='pass' if passed else 'fail',passed=passed,metrics=measured,thresholds=limits)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[str(exc)];return report,2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    except Exception as exc:print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    return code

if __name__=='__main__':sys.exit(main())
