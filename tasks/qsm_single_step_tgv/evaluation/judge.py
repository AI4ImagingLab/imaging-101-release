"""Standalone, solver-independent NPZ evaluator for signed susceptibility maps."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent


def score(prediction, reference, mask):
    a, b = prediction[mask].ravel(), reference[mask].ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not len(a) or nb == 0 or np.ptp(b) == 0:
        raise ValueError('Invalid empty, zero-norm or constant evaluator reference')
    return {'ncc': float(a@b/(na*nb)) if na else 0.,
            'nrmse': float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))}


def observation_metrics(chi, eta, phase, mask):
    """Independent seven-point PDE check; no solver or submitted code import."""
    h = (1.0625, 1.0625, 1.0714285714285714)
    s = 2*np.pi*42.58*2.89362*.025
    c = (slice(1, -1),)*3
    mi = np.zeros_like(mask)
    mi[c] = True
    lphi = np.zeros_like(phase)
    le = np.zeros_like(phase)
    wc = np.zeros_like(phase)
    for axis, step in enumerate(h):
        plus, minus = list(c), list(c)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        plus, minus = tuple(plus), tuple(minus)
        mi[c] &= mask[plus] & mask[minus]
        forward = phase[plus]-phase[c]
        backward = phase[c]-phase[minus]
        # atan2 avoids copying the reconstruction's modulo implementation.
        lphi[c] += (np.arctan2(np.sin(forward), np.cos(forward))-
                    np.arctan2(np.sin(backward), np.cos(backward)))/step**2
        le[c] += (eta[plus]-2*eta[c]+eta[minus])/step**2
        wc[c] += (-1/3, -1/3, 2/3)[axis]*s*(chi[plus]-2*chi[c]+chi[minus])/step**2
    if not mi.any() or np.linalg.norm(lphi[mi]) == 0:
        raise ValueError('Degenerate public phase or evaluation mask')
    return {'pde_relative_residual': float(np.linalg.norm((le-wc-lphi)[mi])/np.linalg.norm(lphi[mi])),
            'eta_rms_rad': float(np.sqrt(np.mean(eta[mask]**2)))}


def evaluate(prediction, asset_dir=HERE):
    result = {'status': 'error', 'passed': False, 'metrics': {}, 'thresholds': {}, 'errors': []}
    try:
        asset_dir = Path(asset_dir)
        cfg = json.loads((asset_dir/'metrics.json').read_text(encoding='utf-8'))
        directions = {key: ('>=' if key.endswith('_ncc') else '<=')
                      for key in ('reconstruction_ncc', 'reconstruction_nrmse',
                                  'residual_phase_ncc', 'residual_phase_nrmse',
                                  'pde_relative_residual', 'eta_rms_rad')}
        boundaries = {key: cfg[key+'_boundary'] for key in directions}
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in boundaries.values()):
            raise ValueError('Invalid nonfinite evaluator boundary')
        if any(v <= 0 or (k.endswith('_ncc') and v > 1) for k, v in boundaries.items()):
            raise ValueError('Invalid or vacuous evaluator boundary')
        result['thresholds'] = {k: {'comparison': directions[k], 'boundary': v} for k, v in boundaries.items()}
        ref_file = asset_dir/'ground_truth.npz'
        if not ref_file.exists():
            ref_file = asset_dir/'baseline_reference.npz'
        with np.load(ref_file, allow_pickle=False) as z:
            refs = {key:z[key] for key in ('reconstruction', 'residual_phase')}
        with np.load(asset_dir.parent/'data/raw_data.npz', allow_pickle=False) as z:
            mask, phase = z['mask'], z['phase']
        if mask.shape != (1, 160, 160, 160) or mask.dtype != bool or phase.shape != mask.shape or not np.isfinite(phase).all():
            raise ValueError('Invalid evaluator public observations')
        for key, ref in refs.items():
            if ref.shape != mask.shape or ref.dtype != np.float64 or not np.isfinite(ref).all():
                raise ValueError('Invalid evaluator reference: '+key)
        with np.load(Path(prediction), allow_pickle=False) as z:
            arrays = {key:z[key] for key in refs}
        for key, a in arrays.items():
            if a.shape != mask.shape:
                raise ValueError(f'Expected {key} shape {mask.shape}, got {a.shape}')
            if a.dtype != np.float64 or np.iscomplexobj(a):
                raise ValueError('Expected real float64 '+key)
            if not np.isfinite(a).all():
                raise ValueError('Prediction contains NaN or Infinity: '+key)
            if np.any(a[~mask] != 0):
                raise ValueError('Prediction must be exactly zero outside the public mask: '+key)
            if abs(float(np.mean(a[mask]))) > 1e-8:
                raise ValueError('Prediction violates the zero brain-mean gauge: '+key)
        measured = {}
        for key in refs:
            measured.update({key+'_'+k:v for k,v in score(arrays[key], refs[key], mask).items()})
        measured.update(observation_metrics(arrays['reconstruction'][0], arrays['residual_phase'][0], phase[0], mask[0]))
        if not all(np.isfinite(value) for value in measured.values()):
            raise ValueError('Nonfinite computed metrics from numerical overflow')
        passed = all(measured[k] >= v if directions[k] == '>=' else measured[k] <= v for k,v in boundaries.items())
        result.update(status='pass' if passed else 'fail', passed=passed, metrics=measured)
        return result, 0 if passed else 1
    except (OSError, KeyError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        return result, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    result, code = evaluate(args.prediction)
    try:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except (OSError, ValueError) as exc:
        print(f'Cannot write evaluation report: {exc}', file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
