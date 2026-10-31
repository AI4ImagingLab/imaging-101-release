"""Independent standalone evaluator for joint spectral material tomography."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

SHAPES = {'reconstruction': (1, 4, 255, 255), 'spectra': (1, 4, 65),
          'dictionary_weights': (1, 4, 4)}


def ray_integrals(images, source, detector):
    """Independent double-precision Siddon integration through square pixels."""
    direction = np.asarray(detector) - source
    edges = np.arange(256, dtype=np.float64) - 127.5
    cuts = [np.array([0., 1.])]
    for axis in range(2):
        if direction[axis] != 0:
            cuts.append((edges - source[axis]) / direction[axis])
    times = np.unique(np.concatenate(cuts))
    times = times[(times >= 0) & (times <= 1)]
    middle = (times[:-1] + times[1:]) * .5
    coordinates = source[:, None] + direction[:, None] * middle
    inside = np.all((coordinates > -127.5) & (coordinates < 127.5), axis=0)
    columns = np.floor(coordinates[0, inside] + 127.5).astype(int)
    rows = np.floor(127.5 - coordinates[1, inside]).astype(int)
    lengths = np.diff(times)[inside] * np.linalg.norm(direction)
    return images[:, rows, columns] @ lengths


def matched_line_integrals(images, source, detector):
    """Selected float32 line discretization, independently evaluated ray by ray."""
    sx, sy = source
    dx, dy = detector
    rx, ry = sx - dx, sy - dy
    half = np.float32(127)
    vertical = abs(rx) < abs(ry)
    if vertical:
        slope = rx / ry
        length = np.sqrt(rx * rx + ry * ry) / abs(ry)
        start = dx + (half - dy) * slope + half
    else:
        slope = ry / rx
        length = np.sqrt(rx * rx + ry * ry) / abs(rx)
        start = -(dy + (-half - dx) * slope - half)
    increments = np.full(255, -slope, dtype=np.float32)
    increments[0] = start
    coordinate = np.add.accumulate(increments, dtype=np.float32)
    nearest = np.floor(coordinate + np.float32(.5)).astype(int)
    offset = coordinate - nearest.astype(np.float32)
    lower = np.float32(.5) - np.float32(.5) * abs(slope)
    upper = np.float32(.5) + np.float32(.5) * abs(slope)
    left, right = offset < -lower, offset > lower
    central = np.full(255, length, dtype=np.float32)
    if slope != 0:
        factor = length / (upper - lower)
        central[left] = (offset[left] + upper) * factor
        central[right] = length - (offset[right] - lower) * factor
        neighbor_weight = length - central
        neighbor_weight[right] = (offset[right] - lower) * factor
    else:
        neighbor_weight = np.zeros(255, dtype=np.float32)
    result = np.zeros(len(images))
    along = np.arange(255)
    for transverse, weights, active in ((nearest, central, np.ones(255, dtype=bool)),
        (np.where(left, nearest - 1, nearest + 1), neighbor_weight, left | right)):
        keep = active & (transverse >= 0) & (transverse < 255)
        rows, columns = (along[keep], transverse[keep]) if vertical else (transverse[keep], along[keep])
        result += images[:, rows, columns] @ weights[keep].astype(np.float64)
    return result


def project_maps(images, angles):
    """Full257-detector matched rays at specified public reconstruction angles."""
    pixel_mm = .80 / (883.081116 / 57.999001)
    source_distance = 57.999001 / pixel_mm
    detector_distance = (883.081116 - 57.999001) / pixel_mm
    step = .80 / pixel_mm
    shift = 257 / 2 - 127.379570
    result = np.empty((len(angles), 257, len(images)), dtype=np.float64)
    for view, angle in enumerate(angles):
        sine, cosine = np.sin(angle), np.cos(angle)
        source = (source_distance * np.array([sine, -cosine])).astype(np.float32)
        detector = detector_distance * np.array([-sine, cosine])
        offset = step * np.array([cosine, sine])
        center = detector + shift * offset
        pixel_step = offset.astype(np.float32)
        start = np.array([np.float32(center[k] - 257 * .5 * float(pixel_step[k])) for k in range(2)])
        for channel in range(257):
            position = start + np.float32(channel + .5) * pixel_step
            result[view, channel] = matched_line_integrals(images, source, position)
    return result


def observation_consistency(prediction, all_views=False):
    """Fixed12-view/all-detector/all-selected-energy public-data consistency."""
    raw = Path(__file__).resolve().parents[1] / 'data' / 'raw_data.npz'
    views = np.arange(600) if all_views else np.arange(0, 600, 50)
    with np.load(raw, allow_pickle=False) as data:
        counts = data['counts'][0, 599 - views].astype(np.float64)
        dark, flat = data['dark'][0].astype(np.float64), data['flat'][0].astype(np.float64)
        angles = data['angles'][0, views]
    if counts.shape != (len(views), 128, 257) or dark.shape != (128, 257) or flat.shape != dark.shape:
        raise ValueError('evaluator public observation shapes are invalid')
    if np.any(counts <= dark) or np.any(flat <= dark):
        raise ValueError('evaluator public dark-corrected counts are invalid')
    measured = np.log((flat - dark) / (counts - dark))[:, 29:94].transpose(0, 2, 1)
    predicted = project_maps(prediction['reconstruction'][0], angles) @ prediction['spectra'][0]
    norm = np.linalg.norm(measured)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError('evaluator observations have degenerate norm')
    return float(np.linalg.norm(predicted - measured) / norm)


def load_arrays(path):
    if not Path(path).is_file():
        raise ValueError(f'NPZ does not exist: {path}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {}
        for key, shape in SHAPES.items():
            if key not in archive:
                raise ValueError(f'missing required key: {key}')
            value = archive[key]
            if value.shape != shape:
                raise ValueError(f'{key}: expected shape {shape}, got {value.shape}')
            if value.dtype != np.dtype('float64'):
                raise ValueError(f'{key}: expected real float64, got {value.dtype}')
            if not np.all(np.isfinite(value)):
                raise ValueError(f'{key}: NaN or Infinity is not allowed')
            arrays[key] = value.copy()
    return arrays


def component_order(prediction, reference):
    norm_x, norm_r = np.linalg.norm(prediction, axis=1), np.linalg.norm(reference, axis=1)
    if np.any(norm_r == 0):
        raise ValueError('evaluator reference has a zero component spectrum')
    similarity = np.divide(prediction @ reference.T, norm_x[:, None] * norm_r[None, :],
                           out=np.zeros((4, 4)), where=norm_x[:, None] != 0)
    rows, columns = linear_sum_assignment(1 - similarity)
    order = np.empty(4, dtype=int)
    order[columns] = rows
    return order


def score_arrays(prediction, reference):
    order = component_order(prediction['spectra'][0], reference['spectra'][0])
    metrics = {}
    for key in SHAPES:
        for component in range(4):
            actual = prediction[key][0, order[component]].ravel()
            target = reference[key][0, component].ravel()
            norm_target, dynamic = np.linalg.norm(target), np.ptp(target)
            if norm_target == 0 or dynamic == 0:
                raise ValueError(f'evaluator reference {key}/{component} is degenerate')
            norm_actual = np.linalg.norm(actual)
            ncc = float(np.dot(actual, target) / (norm_actual * norm_target)) if norm_actual else 0.0
            nrmse = float(np.sqrt(np.mean((actual - target) ** 2)) / dynamic)
            metrics[f'{key}_{component}_ncc'] = ncc
            metrics[f'{key}_{component}_nrmse'] = nrmse
    maps, weights = prediction['reconstruction'][0], prediction['dictionary_weights'][0]
    metrics['nonnegative_violation'] = float(max(0., -min(np.min(v) for v in prediction.values())))
    metrics['abundance_sum_violation'] = float(max(0., np.max(maps.sum(axis=0) - 1)))
    metrics['weight_row_sum_error'] = float(np.max(abs(weights.sum(axis=1) - 1)))
    metrics['weight_column_sum_violation'] = float(max(0., np.max(weights.sum(axis=0) - 1)))
    return metrics


def thresholds_from_config(config):
    thresholds = {}
    for metric in ('nonnegative_violation', 'abundance_sum_violation',
                   'weight_row_sum_error', 'weight_column_sum_violation', 'forward_relative_l2', 'dictionary_relative_l2', 'full_objective_ratio'):
        boundary = config.get(metric + '_boundary')
        if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not np.isfinite(boundary) or boundary < 0:
            raise ValueError(f'undefined or invalid evaluator boundary: {metric}')
        thresholds[metric] = {'comparison': '<=', 'boundary': boundary}
    return thresholds


def evaluate(prediction_path, asset_dir=None):
    report = {'status': 'error', 'passed': False, 'metrics': {}, 'thresholds': {}, 'errors': []}
    try:
        assets = Path(asset_dir) if asset_dir is not None else Path(__file__).resolve().parent
        config_path = assets / 'metrics.json'
        if not config_path.is_file():
            raise ValueError('evaluator metrics.json is unavailable')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        thresholds = thresholds_from_config(config)
        reference_path = assets / 'ground_truth.npz'
        if not reference_path.is_file():
            reference_path = assets / 'baseline_reference.npz'
        reference = load_arrays(reference_path)
        prediction = load_arrays(prediction_path)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            maps=prediction['reconstruction'][0];weights=prediction['dictionary_weights'][0]
            metrics={'nonnegative_violation':float(max(0.,-min(np.min(v) for v in prediction.values()))),
                     'abundance_sum_violation':float(max(0.,np.max(maps.sum(axis=0)-1))),
                     'weight_row_sum_error':float(np.max(abs(weights.sum(axis=1)-1))),
                     'weight_column_sum_violation':float(max(0.,np.max(weights.sum(axis=0)-1)))}
            metrics['forward_relative_l2'] = observation_consistency(prediction)
            with np.load(reference_path,allow_pickle=False) as cached_basis:
                basis=cached_basis['spectral_dictionary'][0] if 'spectral_dictionary' in cached_basis else np.linalg.solve(reference['dictionary_weights'][0],reference['spectra'][0])
            if not np.isfinite(basis).all():raise ValueError('Invalid fixed observation-derived dictionary')
            metrics['dictionary_relative_l2']=float(np.linalg.norm(prediction['spectra'][0]-prediction['dictionary_weights'][0]@basis)/np.linalg.norm(basis))
            full=observation_consistency(prediction,all_views=True)
            standard=observation_consistency(reference,all_views=True)
            if standard<=0 or not np.isfinite(standard):raise ValueError('Degenerate standard objective')
            metrics['full_objective_ratio']=float((full/standard)**2)
            metrics['full_forward_relative_l2']=float(full)
            metrics['standard_full_forward_relative_l2']=float(standard)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError('prediction magnitude caused nonfinite evaluation arithmetic')
        metrics = {name: metrics[name] for name in thresholds}
        passed = all(metrics[name] >= item['boundary'] if item['comparison'] == '>='
                     else metrics[name] <= item['boundary'] for name, item in thresholds.items())
        report.update(status='pass' if passed else 'fail', passed=bool(passed),
                      metrics=metrics, thresholds=thresholds)
        return report, 0 if passed else 1
    except Exception as exc:
        report['errors'] = [f'{type(exc).__name__}: {exc}']
        return report, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    report, code = evaluate(args.prediction)
    try:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except Exception as exc:
        print(f'Cannot write evaluation report: {exc}', file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
