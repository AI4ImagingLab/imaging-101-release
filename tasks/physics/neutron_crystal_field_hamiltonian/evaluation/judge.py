"""Independent finite-domain CEF evaluator; never imports the submitted solver."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import wofz

HERE = Path(__file__).resolve().parent
CONTRACT = {'reconstruction': ((1, 2, 15), 'float64'),
            'neutron_spectrum': ((1, 2, 3, 180), 'float64'),
            'resolution_coefficients': ((1, 2), 'float64'),
            'selected_model': ((1,), 'int64')}

def load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        result = {}
        for key, (shape, dtype) in CONTRACT.items():
            if key not in z: raise ValueError('Missing required key: ' + key)
            a = z[key]
            if a.shape != shape: raise ValueError('Wrong shape for ' + key)
            if a.dtype != np.dtype(dtype): raise ValueError('Wrong dtype for ' + key)
            if not np.all(np.isfinite(a)): raise ValueError('Non-finite values in ' + key)
            result[key] = a.copy()
        return result

def angular_operators():
    j = 7.5
    m = np.arange(-j, j + 1)
    plus = np.zeros((16, 16))
    plus[np.arange(1, 16), np.arange(15)] = np.sqrt((j-m[:-1])*(j+m[:-1]+1))
    minus = plus.T
    x, y, z = (plus+minus)/2, (plus-minus)/(2j), np.diag(m)
    a = j*(j+1)
    eye = np.eye(16)
    z2, z3, z4, z6 = (np.linalg.matrix_power(z, k) for k in (2, 3, 4, 6))
    p3 = np.linalg.matrix_power(plus, 3) + np.linalg.matrix_power(minus, 3)
    q = 11*z3-(3*a+59)*z
    ops = np.array([3*z2-a*eye,
        35*z4-(30*a-25)*z2+(3*a*a-6*a)*eye,
        (p3@z+z@p3)/4,
        231*z6-(315*a-735)*z4+(105*a*a-525*a+294)*z2+(-5*a**3+40*a*a-60*a)*eye,
        (p3@q+q@p3)/4,
        (np.linalg.matrix_power(plus,6)+np.linalg.matrix_power(minus,6))/2])
    return ops, (x, y, z)

def independent_forward(parameters, width, data):
    ops, angular = angular_operators()
    h = np.einsum('i,ijk->jk', parameters[:6], ops)
    energies, v = np.linalg.eigh(h)
    energies -= energies.min()
    strength = sum(np.abs(v.conj().T@op@v)**2 for op in angular)
    result = []
    for k, temperature in enumerate(data['temperature'][0]):
        x = data['energy'][0,k]*(1+temperature*parameters[14])-parameters[10]
        weights = np.exp(-energies/(.0861733*temperature))
        weights /= weights.sum()
        curve = np.zeros_like(x)
        for i in np.flatnonzero(weights > 1e-3):
            delta = energies-energies[i]
            sigma = (width[0]*delta+width[1])/(2*np.sqrt(2*np.log(2)))
            if np.any(np.abs(sigma)<1e-14): raise ValueError('Singular Gaussian width')
            argument = ((x[:,None]-delta)+.5j*parameters[7+k])/(sigma*np.sqrt(2))
            profiles = np.real(wofz(argument))/(sigma*np.sqrt(2*np.pi))
            curve += weights[i]*(profiles@strength[i])
        result.append(parameters[6]*curve+parameters[11+k])
    moments = []
    for axis in (0, 2):
        e, states = np.linalg.eigh(h + 1.2*.057883818012*angular[axis])
        w = np.exp(-(e-e.min())/(.086173303*.5)); w /= w.sum()
        expectation = np.real(np.diag(states.conj().T@angular[axis]@states))
        moments.append(float(-1.2*np.dot(w, expectation)))
    return np.array(result), energies[::2][1:5], moments

def score(prediction, assets=HERE):
    config = json.loads((assets/'metrics.json').read_text(encoding='utf-8'))
    for key, shape in (('parameter_relative_boundary',(2,)),('spectrum_nrmse_boundary',(2,)),
                       ('observation_nrmse_boundary',(2,3)),('forward_consistency_boundary',())):
        boundary=np.asarray(config[key],dtype=float)
        if boundary.shape!=shape or not np.all(np.isfinite(boundary)) or np.any(boundary<=0):
            raise ValueError('Invalid metric boundary configuration: '+key)
    truth = assets/'ground_truth.npz'
    reference = load_prediction(truth if truth.is_file() else assets/'baseline_reference.npz')
    with np.load(assets.parent/'data/raw_data.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    with np.load(assets/'baseline_reference.npz', allow_pickle=False) as z:
        measured_peaks = z['measured_inelastic_peaks_meV']
        if measured_peaks.shape!=(1,4) or measured_peaks.dtype!=np.dtype('float64') or not np.isfinite(measured_peaks).all():
            raise ValueError('Invalid hidden peak measurement contract')
        peaks = measured_peaks[0].copy()
    measured, thresholds = {}, {}
    def metric(key, value, direction, boundary):
        value = float(value)
        if not np.isfinite(value): raise ValueError('Undefined metric: '+key)
        measured[key] = value
        thresholds[key] = {'comparison':direction, 'boundary':float(boundary)}
    p = prediction['reconstruction'][0].copy()
    for row, sign in enumerate((-1, 1)):
        if p[row,2]*sign < 0: p[row,[2,4]] *= -1
    for row in range(2):
        label = 'model'+str(row)
        r = reference['reconstruction'][0,row]
        metric(label+'_parameter_relative_max', np.max(np.abs(p[row]-r)/np.abs(r)), '<=', config['parameter_relative_boundary'][row])
        curve = prediction['neutron_spectrum'][0,row]
        refcurve = reference['neutron_spectrum'][0,row]
        dynamic = np.ptp(refcurve)
        metric(label+'_spectrum_nrmse', np.sqrt(np.mean((curve-refcurve)**2))/dynamic, '<=',config['spectrum_nrmse_boundary'][row])
        physical = np.all(p[row,6:10]>0) and prediction['resolution_coefficients'][0,1]>0 and np.all(curve>=0)
        metric(label+'_physical_domain', physical, '>=', 1)
        forward, excitation, moments = independent_forward(p[row], prediction['resolution_coefficients'][0], data)
        metric(label+'_forward_consistency',np.max(np.abs(curve-forward))/dynamic,'<=',config['forward_consistency_boundary'])
        metric(label+'_excitation_error_meV',np.max(np.abs(excitation-peaks)),'<=',.1)
        metric(label+'_anisotropy_order', (moments[0]>moments[1]) if row==0 else (moments[0]<moments[1]),'>=',1)
        for k in range(3):
            mask=data['fit_mask'][0,k]
            y=data['intensity'][0,k,mask]; yh=curve[k,mask]
            metric(label+'_temperature'+str(k)+'_observed_nrmse',np.sqrt(np.mean((yh-y)**2))/np.ptp(y),'<=',config['observation_nrmse_boundary'][row][k])
            norm = np.linalg.norm(yh)*np.linalg.norm(y)
            metric(label+'_temperature'+str(k)+'_observed_ncc',np.dot(yh,y)/norm if norm else 0.,'>=',.99)
    for i, name in enumerate(('resolution_slope_abs','resolution_intercept_abs_meV')):
        metric(name,abs(prediction['resolution_coefficients'][0,i]-reference['resolution_coefficients'][0,i]),'<=',5e-6)
    ab=(data['field_ab'][0]>=.8)&(data['field_ab'][0]<=1.2)
    c=(data['field_c'][0]>=.8)&(data['field_c'][0]<=1.2)
    selected = 0 if data['moment_ab'][0,ab].mean()>data['moment_c'][0,c].mean() else 1
    metric('selected_model_matches_observation', prediction['selected_model'][0]==selected, '>=',1)
    passed = all(measured[k]<=v['boundary'] if v['comparison']=='<=' else measured[k]>=v['boundary'] for k,v in thresholds.items())
    return {'status':'pass' if passed else 'fail','passed':bool(passed),'metrics':measured,'thresholds':thresholds,'errors':[]}

def evaluate(path, assets=HERE):
    try:
        prediction=load_prediction(path)
        report=score(prediction, assets)
        return report, 0 if report['passed'] else 1
    except Exception as error:
        return {'status':'error','passed':False,'metrics':{},'thresholds':{},'errors':[str(error)]}, 2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report, code=evaluate(args.prediction)
    try:
        target=Path(args.report); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n',encoding='utf-8')
    except Exception as error:
        print('Cannot write report: '+str(error),file=sys.stderr)
        return 2
    return code

if __name__=='__main__': sys.exit(main())
