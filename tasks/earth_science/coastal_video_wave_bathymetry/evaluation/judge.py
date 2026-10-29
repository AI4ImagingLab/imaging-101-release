"""Independent survey/coverage Judge; never imports submitted or reference solvers."""
from pathlib import Path
import argparse,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent
SHAPE=(1,8963)
METRICS={'interior_ncc':'>=','interior_nrmse':'<=','all_valid_ncc':'>=','all_valid_nrmse':'<=','required_coverage':'>=','interior_coverage':'>='}

def load_prediction(path):
    path=Path(path)
    if not path.is_file():raise ValueError('Prediction file is missing: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if len(z.files)!=2 or set(z.files)!={'reconstruction','validity'}:raise ValueError('Expected exactly reconstruction and validity')
        x=z['reconstruction'];valid=z['validity']
        if x.shape!=SHAPE or valid.shape!=SHAPE:raise ValueError('Both output arrays must have shape (1,8963)')
        if x.dtype!=np.dtype('float64') or valid.dtype!=np.dtype('bool'):raise ValueError('Expected real float64 reconstruction and boolean validity')
        if not np.isfinite(x).all():raise ValueError('Nonfinite bed elevation')
        if np.any(x[~valid]!=0):raise ValueError('Invalid points must store the exact zero sentinel')
        return x[0].copy(),valid[0].copy()

def score(x,y):
    """Uncentered cosine and dynamic-range NRMSE, with stable finite arithmetic."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.shape!=y.shape or x.ndim!=1 or not x.size:raise ValueError('Metric expects equal nonempty vectors')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite metric input')
    span=float(np.ptp(y));sx=float(np.max(np.abs(x)));sy=float(np.max(np.abs(y)))
    if not np.isfinite(span) or span<=0 or sy<=0:raise ValueError('Degenerate evaluator reference')
    a=x/sx if sx else np.zeros_like(x);b=y/sy
    norm=float(np.sqrt(np.sum(a*a)*np.sum(b*b)))
    ncc=float(np.clip(np.sum(a*b)/norm,-1.,1.)) if norm else 0.
    common=max(sx,sy);scaled_rms=float(np.sqrt(np.mean((x/common-y/common)**2)))
    if not scaled_rms:nrmse=0.
    else:
        logarithm=np.log(scaled_rms)+np.log(common)-np.log(span)
        nrmse=float(np.exp(logarithm)) if logarithm<np.log(np.finfo(float).max) else float(np.finfo(float).max)
    return ncc,nrmse

def load_assets(assets):
    assets=Path(assets);config=json.loads((assets/'metrics.json').read_text(encoding='utf8'))
    if config.get('comparisons')!=METRICS:raise ValueError('Invalid metric comparison configuration')
    if any(isinstance(config[k+'_boundary'],bool) for k in METRICS):raise ValueError('Boolean boundary is invalid')
    bounds={k:float(config[k+'_boundary']) for k in METRICS}
    if any(not np.isfinite(v) for v in bounds.values()):raise ValueError('Nonfinite boundary')
    if any(not -1<=bounds[k]<=1 for k in ['interior_ncc','all_valid_ncc']):raise ValueError('NCC boundary outside [-1,1]')
    if any(bounds[k]<=0 for k in ['interior_nrmse','all_valid_nrmse']):raise ValueError('NRMSE boundary must be positive')
    if not 0<bounds['required_coverage']<=1:raise ValueError('Invalid coverage boundary')
    if bounds['interior_coverage']!=1:raise ValueError('Interior coverage must be complete')
    # This bound task has independent measured truth. A missing truth asset is
    # an evaluator error, never permission to relabel the method output as GT.
    with np.load(assets/'ground_truth.npz',allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Invalid survey keys')
        y=z['reconstruction']
        if y.shape!=SHAPE or y.dtype!=np.dtype('float64') or not np.isfinite(y).all():raise ValueError('Invalid independent survey array')
        y=y[0].copy()
    with np.load(assets/'baseline_reference.npz',allow_pickle=False) as z:
        if set(z.files)!={'reconstruction','validity','interior_validity'}:raise ValueError('Invalid support asset keys')
        required=z['validity'];interior=z['interior_validity'];baseline=z['reconstruction']
        if required.shape!=SHAPE or interior.shape!=SHAPE or required.dtype!=np.dtype('bool') or interior.dtype!=np.dtype('bool'):raise ValueError('Invalid support masks')
        if baseline.shape!=SHAPE or baseline.dtype!=np.dtype('float64') or not np.isfinite(baseline).all() or np.any(baseline[~required]!=0):raise ValueError('Invalid baseline array')
        required=required[0].copy();interior=interior[0].copy()
    if int(required.sum())!=int(config['required_site_count']) or required.sum()<2:raise ValueError('Invalid measurement-supported footprint')
    if int(interior.sum())!=int(config['interior_site_count']) or interior.sum()<2 or np.any(interior&~required):raise ValueError('Invalid fixed interior footprint')
    if np.ptp(y[required])<=0 or np.ptp(y[interior])<=0:raise ValueError('Degenerate required survey values')
    return y,required,interior,bounds

def evaluate(prediction,assets=HERE):
    report={'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[]}
    try:
        x,valid=load_prediction(prediction);y,required,interior,bounds=load_assets(assets)
        coverage=float(np.count_nonzero(valid&required)/np.count_nonzero(required))
        interior_coverage=float(np.count_nonzero(valid&interior)/np.count_nonzero(interior))
        # Missing interior estimates fail coverage without interpreting their
        # zero storage sentinels as physical elevations or selecting a subset.
        if interior_coverage==1:ncc,err=score(x[interior],y[interior])
        else:ncc,err=0.,float(np.finfo(float).max)
        # Empty/single-valued submitted support is a valid but unsuccessful
        # submission, not an undefined JSON metric. It cannot meet coverage.
        if valid.sum()<2 or np.ptp(y[valid])<=0:ancc,aerr=0.,float(np.finfo(float).max)
        else:ancc,aerr=score(x[valid],y[valid])
        measured={'interior_ncc':ncc,'interior_nrmse':err,'all_valid_ncc':ancc,'all_valid_nrmse':aerr,'required_coverage':coverage,'interior_coverage':interior_coverage}
        report['metrics']=measured
        report['thresholds']={k:{'comparison':METRICS[k],'boundary':bounds[k]} for k in METRICS}
        passed=all(measured[k]>=bounds[k] if METRICS[k]=='>=' else measured[k]<=bounds[k] for k in METRICS)
        report.update(status='pass' if passed else 'fail',passed=bool(passed))
    except Exception as e:
        report.update(status='error',passed=False,metrics={},thresholds={},errors=[str(e)])
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report=evaluate(args.prediction)
    try:
        path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as e:
        print('Report write failed: '+str(e),file=sys.stderr);return 2
    return 0 if report['passed'] else 1 if report['status']=='fail' else 2

if __name__=='__main__':raise SystemExit(main())
