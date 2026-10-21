"""Standalone physical tensor evaluator; no solver imports or spatial alignment."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ASSETS=Path(__file__).resolve().parent
SHAPE=(1,3,50,50)
COMPONENTS=('xx','yy','xy')
def metrics(prediction,reference):
    result={}
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        for index,name in enumerate(COMPONENTS):
            x=prediction[0,index].ravel();r=reference[0,index].ravel()
            span=np.ptp(r);rn=np.linalg.norm(r);xn=np.linalg.norm(x)
            if span<=0 or rn<=0:raise ValueError('Degenerate hidden component '+name)
            result[name+'_ncc']=float(np.dot(x,r)/(xn*rn)) if xn>0 else 0.
            result[name+'_nrmse']=float(np.sqrt(np.mean((x-r)**2))/span)
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metric')
    return result
def read_thresholds(configuration,required_metrics):
    """Read each canonical named boundary exactly once; directions are separate."""
    comparisons=configuration["metric_comparisons"]
    if set(comparisons)!=set(required_metrics):raise ValueError("Undefined metric comparison configuration")
    named={key[:-9] for key in configuration if key.endswith("_boundary")}
    if named!=set(required_metrics):raise ValueError("Missing or unexpected named metric boundary")
    thresholds={}
    for metric in required_metrics:
        comparison=comparisons[metric];boundary=configuration[metric+"_boundary"]
        expected=">=" if metric.endswith("_ncc") else "<="
        if comparison!=expected or not np.isfinite(boundary):raise ValueError("Invalid named boundary "+metric)
        thresholds[metric]={"comparison":comparison,"boundary":boundary}
    return thresholds

def evaluate(path):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        configuration=json.loads((ASSETS/'metrics.json').read_text(encoding='utf-8'));thresholds=read_thresholds(configuration,['xx_ncc', 'xx_nrmse', 'xy_ncc', 'xy_nrmse', 'yy_ncc', 'yy_nrmse'])
        if set(thresholds)!={c+'_'+m for c in COMPONENTS for m in ['ncc','nrmse']}:raise ValueError('Undefined metric configuration')
        for key,item in thresholds.items():
            expected='>=' if key.endswith('_ncc') else '<='
            if item['comparison']!=expected or not np.isfinite(item['boundary']):raise ValueError('Invalid boundary '+key)
        with np.load(ASSETS/'ground_truth.npz',allow_pickle=False) as f:reference=f['reconstruction'].copy()
        if reference.shape!=SHAPE or reference.dtype!=np.float64 or not np.isfinite(reference).all():raise ValueError('Invalid truth shape/dtype/finiteness')
        metrics(reference,reference)
        report['thresholds']=thresholds
    except Exception as exc:
        report['errors']=['Evaluator assets: '+str(exc)];return report,2
    try:
        with np.load(path,allow_pickle=False) as f:
            if 'reconstruction' not in f:raise ValueError('Missing reconstruction key')
            prediction=f['reconstruction'].copy()
        if prediction.shape!=SHAPE:raise ValueError('Expected shape '+str(SHAPE))
        if prediction.dtype!=np.float64:raise ValueError('Expected real float64 stress')
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction')
        values=metrics(prediction,reference)
    except Exception as exc:
        report['errors']=['Prediction: '+str(exc)];return report,2
    report['metrics']=values
    for key,item in thresholds.items():
        if not (values[key]>=item['boundary'] if item['comparison']=='>=' else values[key]<=item['boundary']):report['errors'].append(key+' outside boundary')
    report['passed']=not report['errors'];report['status']='pass' if report['passed'] else 'fail'
    return report,0 if report['passed'] else 1
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except Exception as exc:
        print('Cannot write report: '+str(exc),file=sys.stderr);return 2
    print(report['status']);return code
if __name__=='__main__':sys.exit(main())
