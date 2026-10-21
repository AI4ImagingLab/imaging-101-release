"""Standalone evaluator; imports neither the submitted nor reference solver."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

EXPECTED = {
    'reconstruction': (1, 98, 450, 600),
    'dem_error': (1, 450, 600),
    'deformation_parameters': (1, 5, 450, 600),
    'temporal_coherence': (1, 450, 600),
    'network_timeseries': (1, 98, 450, 600),
}


def validate_arrays(path):
    """Read the exact five float32 arrays; reject malformed/missing submissions."""
    if not Path(path).is_file():
        raise ValueError(f'NPZ file does not exist: {path}')
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for key, shape in EXPECTED.items():
            if key not in archive:
                raise ValueError(f'Missing required array: {key}')
            value = archive[key]
            if value.shape != shape:
                raise ValueError(f'{key}: expected shape {shape}, got {value.shape}')
            if value.dtype != np.dtype('float32'):
                raise ValueError(f'{key}: expected real float32, got {value.dtype}')
            if not np.isfinite(value).all():
                raise ValueError(f'{key}: NaN or infinity')
            arrays[key] = value[0]
    return arrays


def score_pair(estimate, reference):
    """Cosine NCC and RMS error normalized by reference dynamic range."""
    x, y = np.asarray(estimate, np.float64).ravel(), np.asarray(reference, np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid metric arrays')
    span = float(np.max(y)-np.min(y)); norm = float(np.linalg.norm(y)); xnorm = float(np.linalg.norm(x))
    if span <= 0 or norm <= 0:
        raise ValueError('Degenerate reference requires a different metric configuration')
    ncc = float(np.dot(x, y)/(xnorm*norm)) if xnorm else 0.
    nrmse = float(np.sqrt(np.mean((x-y)**2))/span)
    return ncc, nrmse


def observation_consistency(network, data_path):
    """Circular phase residual coherence on the public all-edge validity mask."""
    with np.load(data_path, allow_pickle=False) as raw:
        mask = raw['analysis_mask'][0]
        pairs = raw['pairs'][0]
        phase = raw['phase'][0]
        ref_phase = phase[:, 76, 156]
        displacement = network[:, mask].astype(np.float64)
        total = np.zeros(int(mask.sum()), np.complex128)
        for edge, (start, end) in enumerate(pairs):
            observed = phase[edge, mask].astype(np.float64)-float(ref_phase[edge])
            model = (-4*np.pi/.05546576)*(displacement[end]-displacement[start])
            total += np.exp(1j*(observed-model))
    return float(np.mean(np.abs(total)/len(pairs)))


def evaluate(prediction, directory=None):
    """Return report and exit code; every quantity and consistency metric must pass."""
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    report = {'status': 'error', 'passed': False, 'metrics': {}, 'thresholds': {}, 'errors': []}
    try:
        config = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        reference_path = directory/'ground_truth.npz'
        if not reference_path.exists():
            reference_path = directory/'baseline_reference.npz'
        reference = validate_arrays(reference_path)
        for key in EXPECTED:
            for metric in ['ncc', 'nrmse']:
                boundary = config[f'{key}_{metric}_boundary']
                if not isinstance(boundary, (int, float)) or not np.isfinite(boundary):
                    raise ValueError(f'Invalid threshold for {key}_{metric}')
        with np.load(directory.parent/'data/raw_data.npz', allow_pickle=False) as raw:
            mask = raw['analysis_mask'][0]
        if mask.shape != (450, 600) or mask.dtype != bool or not mask.any():
            raise ValueError('Invalid public analysis mask')
    except Exception as exc:
        report['errors'] = [f'Evaluator asset error: {exc}']
        return report, 2
    try:
        output = validate_arrays(prediction)
    except Exception as exc:
        report['errors'] = [f'Invalid prediction: {exc}']
        return report, 2
    try:
        passed = True
        for key in EXPECTED:
            ncc, nrmse = score_pair(output[key][..., mask], reference[key][..., mask])
            for metric, value, direction in [('ncc', ncc, '>=',), ('nrmse', nrmse, '<=')]:
                name = f'{key}_{metric}'
                boundary = config[f'{name}_boundary']
                report['metrics'][name] = value
                report['thresholds'][name] = {'comparison': direction, 'boundary': boundary}
                passed &= value >= boundary if direction=='>=' else value <= boundary
        consistency = observation_consistency(output['network_timeseries'], directory.parent/'data/raw_data.npz')
        boundary = config['observation_coherence_boundary']
        if not isinstance(boundary, (int, float)) or not np.isfinite(boundary):
            raise ValueError('Invalid observation threshold')
        report['metrics']['observation_coherence'] = consistency
        report['thresholds']['observation_coherence'] = {'comparison': '>=', 'boundary': boundary}
        passed &= consistency >= boundary
        report.update(status='pass' if passed else 'fail', passed=bool(passed))
        return report, 0 if passed else 1
    except Exception as exc:
        report.update(status='error', passed=False, metrics={}, thresholds={}, errors=[f'Evaluation error: {exc}'])
        return report, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    report, code = evaluate(args.prediction)
    try:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
