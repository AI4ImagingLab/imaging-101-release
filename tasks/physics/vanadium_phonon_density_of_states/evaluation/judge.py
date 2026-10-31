"""Independent public-model scoring; never import the submitted solver."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import RectBivariateSpline

NAMES = ('marginal_relative_l2', 'normalization_abs_error', 'negative_density_max',
         'above_support_max', 'low_energy_debye_abs_error')

def configuration(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Metric configuration must be an object')
    expected = {x + '_boundary' for x in NAMES}
    if {x for x in value if x.endswith('_boundary')} != expected:
        raise ValueError('Metric boundary set does not match the declared contract')
    result = {}
    for name in NAMES:
        number = value[name + '_boundary']
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError('Metric boundaries must be finite nonnegative JSON numbers')
        result[name] = {'comparison': '<=', 'boundary': number}
    if 'comparisons' in value and value['comparisons'] != {x: '<=' for x in NAMES}:
        raise ValueError('Invalid comparison configuration')
    return result

def observation(path):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in ('intensity','variance','measured_mask','q_axis','energy_axis')}
    x, v, mask = data['intensity'], data['variance'], data['measured_mask']
    if x.shape != (1,300,149) or v.shape != x.shape or mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError('Invalid evaluator observation shape or acquisition mask')
    if x.dtype != np.float64 or v.dtype != np.float64 or not np.array_equal(np.isfinite(x),mask):
        raise ValueError('Invalid evaluator observation precision or mask')
    if not np.isfinite(v[mask]).all() or np.any(v[mask] < 0):
        raise ValueError('Invalid evaluator measured variance')
    q, en = data['q_axis'], data['energy_axis']
    if q.shape != (1,300) or en.shape != (1,149) or not np.isfinite(q).all() or not np.isfinite(en).all():
        raise ValueError('Invalid evaluator axes')
    target = np.arange(-15.,80.)
    raw = x[0]; valid = mask[0]
    out = RectBivariateSpline(en[0],q[0],np.where(valid,raw,0.).T,kx=1,ky=1)(target,q[0]).T
    for index, row in enumerate(valid):
        positions = np.flatnonzero(row)
        lo, hi = (en[0,positions[0]],en[0,positions[-1]]) if len(positions) else (en[0,0],en[0,0])
        if lo > -15:
            out[index,:int(np.rint(min(lo,79)+15))+1] = np.nan
        if hi < 79:
            out[index,int(np.rint(max(hi,-15)+15)):] = np.nan
    return out, q[0]

def modeled(g, q):
    """Direct convolution implementation of the disclosed finite-grid model."""
    density = np.pad(g, (0,160))
    density = density / density.sum()
    ep = np.arange(240.)
    beta = 1/(300*.0862)
    integrand = density[1:] / ep[1:] / np.tanh(beta*ep[1:]/2)
    gamma = float(integrand.sum() + 2*integrand[0] - integrand[1])
    energy = np.arange(-239.,240.)
    a1 = np.zeros(479)
    nonzero = energy != 0
    a1[nonzero] = density[np.abs(energy[nonzero]).astype(int)] / (
        energy[nonzero] * gamma * (-np.expm1(-beta*energy[nonzero])))
    a1[239] = max(0.,2*a1[240]-a1[241])
    orders = [a1]
    for _ in range(3):
        spectrum = np.convolve(a1,orders[-1],mode='same')
        orders.append(spectrum / spectrum.sum())
    recoil = 6.2415097e21 * (1.0545716e-34*q/1e-10)**2 / (2*50.94*1.66053873e-27)
    dw = gamma * recoil
    indices = np.arange(224,319)
    single = (dw*np.exp(-dw))[:,None] * a1[indices]
    multiple = np.zeros((300,95))
    for n in range(2,5):
        multiple += (dw**n*np.exp(-dw)/math.factorial(n))[:,None] * orders[n-1][indices]
    factor = np.sqrt(2e-3*1.60217733e-19/1.6749286e-27)/(1.054571628e-34/1.6749286e-27*1e10)
    ki = np.sqrt(120.)*factor
    kf = np.sqrt(120.-np.arange(-15.,80.))*factor
    accessible = ((ki+kf[None]>q[:,None]) & (ki+q[:,None]>kf[None]) & (kf[None]+q[:,None]>ki))
    multiple[~accessible] = 0
    scattering = .2*multiple.sum(axis=0)/accessible.sum(axis=0)
    result = single + multiple + scattering
    result[~accessible] = np.nan
    return result

def score(g, measured, q):
    metrics = {
        'normalization_abs_error': abs(float(g.sum())-1.),
        'negative_density_max': max(0.,-float(g.min())),
        'above_support_max': float(np.max(np.abs(g[56:]))),
        'low_energy_debye_abs_error': float(np.max(np.abs(g[:7]-np.arange(7.)**2*g[7]/49)))
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('Prediction causes overflow in physical constraint evaluation')
    # A normalized nonnegative density is needed for the physical forward domain.
    # Invalid physical candidates fail explicitly without creating undefined ratios.
    if g.sum() <= 0 or g[1:].sum() <= 0 or g.min() < -1e-10:
        return metrics
    model = modeled(np.maximum(g,0),q)
    selected = np.isfinite(measured)
    if not np.isfinite(model[selected]).all():
        raise ValueError('Model is undefined on acquired observations')
    model = np.where(selected,model,np.nan)
    area = np.nansum(measured[:,22:])
    if area <= 0:
        raise ValueError('Invalid evaluator inelastic intensity normalization')
    scaled = measured * (np.nansum(model[:,22:])/area)
    target = np.nansum(scaled[:,22:],axis=0)
    difference = np.nansum(scaled[:,22:]-model[:,22:],axis=0)
    norm = np.linalg.norm(target)
    if norm <= 0:
        raise ValueError('Undefined marginal normalization')
    metrics['marginal_relative_l2'] = float(np.linalg.norm(difference)/norm)
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('Nonfinite evaluation metric')
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction',required=True)
    parser.add_argument('--report',required=True)
    args = parser.parse_args()
    report = dict(status='error',passed=False,metrics={},thresholds={},errors=[])
    code = 2
    root = Path(__file__).resolve().parent
    try:
        report['thresholds'] = configuration(root/'metrics.json')
        measured,q = observation(root.parent/'data/raw_data.npz')
        with np.load(args.prediction,allow_pickle=False) as z:
            if set(z.files) != {'reconstruction'}:
                raise ValueError('Expected exactly the reconstruction output key')
            g = z['reconstruction']
        if g.shape != (1,80) or g.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('Expected real float32/float64 reconstruction shape (1,80)')
        if not np.isfinite(g).all():
            raise ValueError('Prediction contains nonfinite values')
        report['metrics'] = score(g[0].astype(np.float64),measured,q)
        passed = (set(report['metrics']) == set(NAMES) and all(
            report['metrics'][k] <= report['thresholds'][k]['boundary'] for k in report['metrics']))
        report.update(status='pass' if passed else 'fail',passed=passed)
        if set(report['metrics']) != set(NAMES):
            report['errors'].append('Prediction is outside the nonnegative normalized forward-model domain')
        code = 0 if passed else 1
    except Exception as exc:
        report.update(status='error',passed=False,metrics={},errors=[f'{type(exc).__name__}: {exc}'])
    try:
        output = Path(args.report)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}',file=sys.stderr)
        return 2
    return code

if __name__ == '__main__':
    if hasattr(sys.stderr,'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
    raise SystemExit(main())
