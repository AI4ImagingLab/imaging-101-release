"""Standalone law and full-acquisition equilibrium evaluator; no solver imports."""
from pathlib import Path
import argparse,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parent

OPERATOR_SHAPES={'Q':(2189,171),'G':(2189,171,4),'H':(171,171),'b':(171,),'C':()}
REFERENCE_SHAPES={'energy':(1,2189),'stress':(1,2189,2,2)}
MAX_FLOAT=float(np.finfo(np.float64).max)

def load_prediction(path):
    if not Path(path).is_file():raise ValueError('Prediction file missing')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'reconstruction'}:raise ValueError('Exactly reconstruction key is required')
        a=z['reconstruction']
        if a.shape!=(1,171):raise ValueError('reconstruction shape must be (1,171)')
        if a.dtype not in (np.dtype('float32'),np.dtype('float64')):raise ValueError('Real float32 or float64 coefficients required')
        if not np.isfinite(a).all():raise ValueError('Nonfinite coefficients')
        if np.any(a<0):raise ValueError('Coefficients must be nonnegative')
        return a[0].astype(np.float64)

def validated_arrays(path,shapes):
    """Hidden arrays use the fixed acquisition's exact float64 schema."""
    with np.load(path,allow_pickle=False) as z:
        missing=set(shapes)-set(z.files)
        if missing:raise ValueError('Hidden asset missing keys: '+','.join(sorted(missing)))
        arrays={}
        for key,shape in shapes.items():
            a=z[key]
            if a.shape!=shape or a.dtype!=np.dtype('float64'):raise ValueError('Hidden '+key+' requires float64 shape '+str(shape))
            if not np.isfinite(a).all():raise ValueError('Hidden '+key+' contains nonfinite values')
            arrays[key]=a
        return arrays

def load_assets():
    """Actual truth, if present, supplies W/P; calibrated operators stay separate."""
    truth=ROOT/'ground_truth.npz';baseline=ROOT/'baseline_reference.npz'
    reference_path=truth if truth.is_file() else baseline
    reference=validated_arrays(reference_path,REFERENCE_SHAPES)
    # Existing baseline carries acquisition operators, independently of its law.
    operator_path=baseline if baseline.is_file() else reference_path
    operators=validated_arrays(operator_path,OPERATOR_SHAPES)
    if float(operators['C'])<=0:raise ValueError('Hidden force normalization C must be positive')
    for key in ('Q','G','H'):
        if not np.any(operators[key]):raise ValueError('Degenerate hidden operator '+key)
    for key in REFERENCE_SHAPES:
        if not np.any(reference[key]):raise ValueError('Degenerate hidden reference '+key)
    return operators,reference

def finite_from_log(value):
    """Represent a nonnegative metric, saturating only beyond float64 range."""
    if value>=math.log(MAX_FLOAT):return MAX_FLOAT
    return math.exp(value)

def relative_l2(operator,theta,reference):
    """L2(A theta-y)/L2(y), with scaled products and log-sized amplitudes."""
    matrix=np.asarray(operator,dtype=np.float64)
    y=np.asarray(reference,dtype=np.float64).reshape(-1)
    sy=float(np.max(abs(y)));normalized_y=y/sy;ny=float(np.linalg.norm(normalized_y))
    scale=float(np.max(abs(theta)))
    if scale==0:return 1.
    sm=float(np.max(abs(matrix)))
    if sm==0:return 1.
    normalized_prediction=(matrix/sm)@(theta/scale)
    lp=math.log(scale)+math.log(sm);ly=math.log(sy);common=max(lp,ly)
    delta=math.exp(lp-common)*normalized_prediction-math.exp(ly-common)*normalized_y
    error=float(np.linalg.norm(delta))
    if error==0:return 0.
    return finite_from_log(common-ly+math.log(error)-math.log(ny))

def weak_residual_relative_l2(theta,H,b,C):
    """sqrt(V/C), preserving the unnormalized physical quadratic without overflow."""
    scale=float(np.max(abs(theta)));C=float(C)
    if scale==0:return 1.
    # Preserve ordinary-data arithmetic; use the scaled path only when needed.
    hs=float(np.max(abs(H)));bs=float(np.max(abs(b)));n=len(theta)
    bound=2*math.log(scale)+math.log(hs)+2*math.log(n)
    if bound<math.log(MAX_FLOAT)-4:
        weak=float(.5*theta@H@theta-b@theta+C)
        if math.isfinite(weak):
            if weak < -1e-10*C:raise ValueError('Hidden quadratic gives negative squared residual')
            return 0. if weak<=0 else finite_from_log(.5*(math.log(weak)-math.log(C)))
    u=theta/scale
    quadratic=.5*float(u@(H/hs)@u)
    linear=float((b/bs)@u) if bs else 0.
    la=2*math.log(scale)+math.log(hs)
    lb=math.log(scale)+math.log(bs) if bs else -math.inf
    lc=math.log(C);common=max(la,lb,lc)
    weak_scaled=quadratic*math.exp(la-common)-linear*math.exp(lb-common)+math.exp(lc-common)
    if weak_scaled<0:raise ValueError('Hidden quadratic gives negative squared residual')
    if weak_scaled==0:return 0.
    return finite_from_log(.5*(common+math.log(weak_scaled)-lc))

def measure(theta,operators,reference):
    G=operators['G'].transpose(0,2,1).reshape(2189*4,171)
    return dict(energy_relative_l2=relative_l2(operators['Q'],theta,reference['energy']),stress_relative_l2=relative_l2(G,theta,reference['stress']),weak_residual_relative_l2=weak_residual_relative_l2(theta,operators['H'],operators['b'],operators['C']))
def evaluate(prediction):
    report=dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    try:
        theta=load_prediction(prediction)
        operators,reference=load_assets()
        values=measure(theta,operators,reference)
        config=json.loads((ROOT/'metrics.json').read_text(encoding='utf8'))
        for key,value in values.items():
            bound=config[key+'_boundary']
            if isinstance(bound,bool) or not np.isfinite(value) or not np.isfinite(bound) or bound<=0:raise ValueError('Invalid metric or threshold')
            report['thresholds'][key]=dict(comparison='<=',boundary=bound)
        report['metrics']=values; report['passed']=all(v<=report['thresholds'][k]['boundary'] for k,v in values.items())
        report['status']='pass' if report['passed'] else 'fail'
        return report,0 if report['passed'] else 1
    except Exception as error:
        report['metrics']={};report['errors']=[str(error)];return report,2
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prediction',required=True);parser.add_argument('--report',required=True);args=parser.parse_args()
    report,code=evaluate(args.prediction)
    try:
        p=Path(args.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    except Exception as error:print('Report write failed: '+str(error),file=sys.stderr);return 2
    print(json.dumps(report,indent=2,allow_nan=False));return code
if __name__=='__main__':sys.exit(main())
