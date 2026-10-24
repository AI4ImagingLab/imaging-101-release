"""Standalone all-interface/all-cell evaluator; no solver imports or execution."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPES={'tensions':(1,357),'pressures':(1,65)}

def load_field(path,role):
    path=Path(path)
    if not path.is_file():raise ValueError(f'{role} file does not exist: {path}')
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=set(SHAPES):raise ValueError(f'{role}: required keys are tensions and pressures')
        result={}
        for key,shape in SHAPES.items():
            a=data[key]
            if a.shape!=shape:raise ValueError(f'{role} {key}: expected shape {shape}, received {a.shape}')
            if a.dtype.kind!='f' or a.dtype.itemsize not in [4,8]:raise ValueError(f'{role} {key}: expected real float32 or float64')
            if not np.isfinite(a).all():raise ValueError(f'{role} {key}: non-finite values')
            result[key]=a[0].astype(np.float64)
    return result

def reference_metrics(x,r):
    if np.ptp(r)<=0 or np.linalg.norm(r)==0:raise ValueError('Evaluator reference is degenerate')
    norm=np.linalg.norm(x)
    return {'ncc':float(x@r/(norm*np.linalg.norm(r))) if norm>0 else 0.,
            'nrmse':float(np.sqrt(np.mean((x-r)**2))/np.ptp(r))}

def observation_physics(tensions,pressures,raw_path):
    """Independently evaluate discrete pressure normal equations on public geometry.

    Each triangle contributes its explicit analytic area and signed-volume
    derivatives. The normal-equation residual tests the conditional pressure
    optimum; it is not a claim that noisy observed mesh is in exact equilibrium.
    """
    with np.load(raw_path,allow_pickle=False) as raw:
        v=raw['vertices'][0];t=raw['triangles'][0];labels=raw['region_labels'][0]
    interfaces,inv=np.unique(np.sort(labels,axis=1),axis=0,return_inverse=True)
    cells=np.unique(labels)
    if len(interfaces)!=357 or len(cells)!=65 or cells[0]!=0:raise ValueError('Evaluator public geometry has unexpected labels')
    region_index=np.searchsorted(cells,labels)
    xyz=v[t];x,y,z=xyz[:,0],xyz[:,1],xyz[:,2]
    n=np.cross(y-x,z-x);length=np.linalg.norm(n,axis=1)
    if np.any(length<=0):raise ValueError('Evaluator geometry has degenerate triangles')
    n=n/length[:,None]
    ga=np.stack([np.cross(n,z-y),np.cross(n,x-z),np.cross(n,y-x)],axis=1)/2
    gv=np.stack([np.cross(y,z),np.cross(z,x),np.cross(x,y)],axis=1)/6
    force=np.zeros_like(v,dtype=float);pressure_force=np.zeros_like(v,dtype=float)
    np.add.at(force,t.ravel(),(ga*tensions[inv,None,None]).reshape(-1,3))
    dp=pressures[region_index[:,0]]-pressures[region_index[:,1]]
    np.add.at(pressure_force,t.ravel(),(gv*dp[:,None,None]).reshape(-1,3))
    residual=force-pressure_force
    moments=np.einsum('ijk,ijk->i',gv,residual[t]);rhs=np.einsum('ijk,ijk->i',gv,force[t])
    r=np.zeros(65);h=np.zeros(65)
    np.add.at(r,region_index[:,0],moments);np.add.at(r,region_index[:,1],-moments)
    np.add.at(h,region_index[:,0],rhs);np.add.at(h,region_index[:,1],-rhs)
    denom=np.linalg.norm(h[1:])
    normal=float(np.linalg.norm(r[1:])/denom) if denom>0 else float(np.linalg.norm(r[1:]))
    return {'pressure_normal_equations_relative_l2':normal,
            'mean_tension_absolute_error':float(abs(tensions.mean()-1)),
            'medium_pressure_absolute':float(abs(pressures[0]))}

def evaluate(prediction,asset_dir=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        root=Path(asset_dir)
        definition=json.loads((root/'metrics.json').read_text(encoding='utf8'))
        limits=definition['thresholds']
        expected={f'{k}_{m}' for k in SHAPES for m in ['ncc','nrmse']}|{'pressure_normal_equations_relative_l2','mean_tension_absolute_error','medium_pressure_absolute'}
        if set(limits)!=expected:raise ValueError('Evaluator threshold keys are incomplete')
        for name,limit in limits.items():
            if limit['comparison'] not in ['<=','>='] or not np.isfinite(limit['boundary']):raise ValueError('Evaluator has invalid threshold definition')
            expected_comparison='>=' if name.endswith('_ncc') else '<='
            if limit['comparison']!=expected_comparison:raise ValueError('Evaluator has reversed metric comparison')
            if name.endswith('_ncc') and not 0<=limit['boundary']<=1:raise ValueError('Evaluator has invalid NCC boundary')
            if ('nrmse' in name or 'absolute' in name or 'relative_l2' in name) and limit['boundary']<0:raise ValueError('Evaluator has negative error boundary')
            if name.endswith('_nrmse') and limit['boundary']==0:raise ValueError('Evaluator needs a nonzero measured numerical tolerance')
        refpath=root/'ground_truth.npz'
        if not refpath.exists():refpath=root/'baseline_reference.npz'
        ref=load_field(refpath,'reference');estimate=load_field(prediction,'prediction')
        values={}
        for key in SHAPES:
            values.update({key+'_'+m:v for m,v in reference_metrics(estimate[key],ref[key]).items()})
        values.update(observation_physics(estimate['tensions'],estimate['pressures'],root.parent/'data/raw_data.npz'))
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Undefined evaluation metric')
        passed=all(values[k]<=v['boundary'] if v['comparison']=='<=' else values[k]>=v['boundary'] for k,v in limits.items())
        report.update(status='pass' if passed else 'fail',passed=bool(passed),metrics=values,thresholds=limits)
        return report,0 if passed else 1
    except Exception as exc:
        report['errors']=[f'{type(exc).__name__}: {exc}']
        return report,2

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args(argv)
    report,code=evaluate(args.prediction)
    try:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    except Exception as exc:
        print(f'Cannot write report: {exc}',file=sys.stderr);return 2
    return code

if __name__=='__main__':raise SystemExit(main())
