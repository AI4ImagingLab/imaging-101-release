"""Independent pose-v1.0 evaluator; no solver execution or solver imports."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np

N = 432
VERSION = 'pose-v1.0'
BOUNDARIES = {'joint_piece_accuracy_boundary', 'oriented_neighbor_accuracy_boundary'}


def load_pose(path, role):
    if not path.is_file():
        raise ValueError(role + ' file is missing')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 2 or set(archive.files) != {'positions', 'quarter_turns'}:
            raise ValueError(role + ' must contain exactly positions and quarter_turns')
        positions, turns = archive['positions'], archive['quarter_turns']
    for name, array, shape in [('positions', positions, (1, N, 2)),
                               ('quarter_turns', turns, (1, N))]:
        if array.shape != shape:
            raise ValueError(role + ': wrong ' + name + ' shape')
        if array.dtype.kind != 'i' or array.dtype.itemsize != 8:
            raise ValueError(role + ': ' + name + ' must have signed int64 dtype')
        if not np.isfinite(array).all():
            raise ValueError(role + ': nonfinite ' + name)
    p, q = positions[0], turns[0]
    if np.any(p < 0) or not np.array_equal(p.min(axis=0), [0, 0]):
        raise ValueError(role + ': positions must be nonnegative with origin (0,0)')
    extent = (int(p[:, 0].max()) + 1, int(p[:, 1].max()) + 1)
    if extent not in {(18, 24), (24, 18)} or len(np.unique(p, axis=0)) != N:
        raise ValueError(role + ': positions must fill one complete collision-free 18x24 or 24x18 rectangle')
    if np.any(q < 0) or np.any(q > 3):
        raise ValueError(role + ': quarter_turns must lie in {0,1,2,3}')
    return p, q


def load_thresholds(folder):
    with (folder / 'metrics.json').open(encoding='utf-8-sig') as stream:
        config = json.load(stream)
    if config.get('evaluation_version') != VERSION:
        raise ValueError('Unsupported evaluator definition version')
    if {k for k in config if k.endswith('_boundary')} != BOUNDARIES:
        raise ValueError('Missing or undefined metric boundary')
    thresholds = {}
    for key in sorted(BOUNDARIES):
        value = config[key]
        if type(value) not in {float, int} or not math.isfinite(value) or value != 0.95:
            raise ValueError('Metric boundary differs from independently frozen pose-v1.0 standard')
        thresholds[key.removesuffix('_boundary')] = dict(comparison='>=', boundary=float(value))
    return thresholds


def compare_common_gauges(p, q, pref, qref, thresholds):
    delta = pref[None, :, :] - pref[:, None, :]
    seams = np.argwhere((np.abs(delta).sum(axis=2) == 1) &
                        np.triu(np.ones((N, N), dtype=bool), 1))
    if len(seams) != 822:
        raise ValueError('Reference adjacency does not match the declared complete grid')
    first, second = seams.T
    candidates = []
    a = p.copy()
    for k in range(4):
        if k:
            a = np.column_stack((-a[:, 1], a[:, 0]))
            a -= a.min(axis=0)
        b = (q + k) % 4
        correct_position = np.all(a == pref, axis=1)
        correct_orientation = b == qref
        correct_neighbor = np.all(a[second]-a[first] == pref[second]-pref[first], axis=1)
        metrics = dict(
            position_accuracy=float(correct_position.mean()),
            orientation_accuracy=float(correct_orientation.mean()),
            joint_piece_accuracy=float((correct_position & correct_orientation).mean()),
            neighbor_accuracy=float(correct_neighbor.mean()),
            oriented_neighbor_accuracy=float((correct_neighbor & correct_orientation[first] &
                                               correct_orientation[second]).mean()),
        )
        passed = all(metrics[name] >= item['boundary'] for name, item in thresholds.items())
        candidates.append(dict(global_quarter_turn=k, passed=passed, metrics=metrics))
    # Prefer a passing common candidate; ties are deterministic and never align metrics separately.
    chosen = max(candidates, key=lambda x: (x['passed'], x['metrics']['joint_piece_accuracy'],
                 x['metrics']['oriented_neighbor_accuracy'], -x['global_quarter_turn']))
    return chosen, candidates


def evaluate(prediction, folder):
    report = dict(status='error', passed=False, metrics={}, thresholds={}, errors=[],
                  evaluation_version=VERSION)
    try:
        thresholds = load_thresholds(folder)
        # This task has actual known acquisition truth; do not guess another reference.
        pref, qref = load_pose(folder / 'ground_truth.npz', 'Evaluator reference')
        p, q = load_pose(prediction, 'Prediction')
        chosen, candidates = compare_common_gauges(p, q, pref, qref, thresholds)
        report.update(status='pass' if chosen['passed'] else 'fail', passed=chosen['passed'],
                      metrics=chosen['metrics'], thresholds=thresholds,
                      alignment=dict(global_quarter_turn=chosen['global_quarter_turn'],
                                     common_for_all_metrics=True), alignment_candidates=candidates)
        return report, 0 if chosen['passed'] else 1
    except Exception as exc:
        report['errors'] = [type(exc).__name__ + ': ' + str(exc)]
        return report, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result, code = evaluate(args.prediction, Path(__file__).resolve().parent)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    except Exception as exc:
        print('Could not write evaluator report: ' + str(exc), file=sys.stderr)
        return 2
    return code


if __name__ == '__main__':
    raise SystemExit(main())
