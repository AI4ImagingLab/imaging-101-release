"""Standalone submission validation and deterministic complex-volume scoring.

No solver or submitted code is imported. Thresholds and reference assets must
be independently validated before this evaluator can declare a prediction pass.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy import fft, signal, ndimage

HERE = Path(__file__).resolve().parent
CONTRACT = {"reconstruction": ((1, 2, 126, 256, 256), np.dtype("complex128")),
            "coherence": ((1, 2, 16, 16, 16), np.dtype("float64"))}
METRIC_NAMES = ('kernel_mass_error','kernel_negative_mass','raw_amplitude_residual','support_fraction','outside_support_excess')


def validate_thresholds(thresholds):
    required = {f"p{p}_{name}" for p in (1, 2) for name in METRIC_NAMES}
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise ValueError("Threshold names must match every required metric exactly")
    for name, rule in thresholds.items():
        if not isinstance(rule, dict) or set(rule) != {"boundary", "comparison"}:
            raise ValueError(f"Invalid threshold rule: {name}")
        boundary = rule["boundary"]
        if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not np.isfinite(boundary):
            raise ValueError(f"Invalid numeric boundary: {name}")
        metric = name.split("_", 1)[1]
        comparison = ">=" if metric == "complex_cosine" else "<="
        if rule["comparison"] != comparison:
            raise ValueError(f"Invalid comparison direction: {name}")
        if metric == "complex_cosine":
            valid = 0. < boundary <= 1.
        elif metric == "kernel_negative_mass":
            valid = boundary == 0.
        elif metric == "kernel_mass_error":
            valid = 0. <= boundary <= 1e-6
        else:
            valid = 0. <= boundary < 1.
        if not valid:
            raise ValueError(f"Invalid or vacuous boundary range: {name}")


def validate_metric_specification(specification):
    """Require one consistent value for each flat/nested metric boundary."""
    if not isinstance(specification, dict) or "thresholds" not in specification:
        raise ValueError("Metric specification requires explicit threshold rules")
    thresholds = specification["thresholds"]
    validate_thresholds(thresholds)
    expected = {f"{name}_boundary" for name in thresholds}
    actual = {key for key in specification if key.endswith("_boundary")}
    if actual != expected:
        raise ValueError("Flat boundary names must match every nested metric rule exactly")
    for name, rule in thresholds.items():
        boundary = specification[f"{name}_boundary"]
        if (isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or
                not np.isfinite(boundary)):
            raise ValueError(f"Invalid flat numeric boundary: {name}")
        if boundary != rule["boundary"]:
            raise ValueError(f"Flat/nested boundary mismatch: {name}")
    return thresholds


def validate_reference(reference, counts):
    if np.any(counts < 0):
        raise ValueError("Evaluator counts must be nonnegative")
    for position in range(2):
        obj = reference["reconstruction"][0, position]
        kernel = reference["coherence"][0, position]
        norm = float(np.sum(abs(obj)**2))
        if not np.isfinite(norm) or norm <= 0.:
            raise ValueError("Evaluator density reference has invalid or zero norm")
        reference_masks(obj)
        if (not np.isfinite(kernel).all() or np.any(kernel < 0.) or
                abs(float(kernel.sum())-1.) > 1e-8):
            raise ValueError("Evaluator kernel reference must be a finite unit-mass probability")
        if np.sum(counts[position], dtype=np.float64) <= 0.:
            raise ValueError("Evaluator observation has zero photon sum")


def load_arrays(path):
    """Validate the complete output contract before any metric is evaluated."""
    if not Path(path).is_file():
        raise ValueError(f"Missing NPZ file: {path}")
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for key, (shape, dtype) in CONTRACT.items():
            if key not in archive.files:
                raise ValueError(f"Missing required array: {key}")
            value = archive[key]
            if value.shape != shape:
                raise ValueError(f"{key}: expected shape {shape}, got {value.shape}")
            if value.dtype != dtype:
                raise ValueError(f"{key}: expected dtype {dtype}, got {value.dtype}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key}: NaN or Infinity is invalid")
            arrays[key] = value
    return arrays


def align_to_reference(reference, candidate):
    """Only integer cyclic translation, global phase and conjugate inversion."""
    reference_fft = fft.fftn(reference)
    best_score, best = -1., None
    for twin in (False, True):
        other = np.conj(np.flip(candidate)) if twin else candidate
        correlation = fft.ifftn(reference_fft * np.conj(fft.fftn(other)))
        index = np.unravel_index(np.argmax(abs(correlation)), reference.shape)
        value = correlation[index]
        score = float(abs(value))
        if score > best_score:
            phase = value / score if score else 1.
            best = np.roll(other, index, axis=tuple(range(other.ndim))) * phase
            best_score = score
    return best


def relative_l2(estimate, reference):
    denominator = float(np.sum(abs(reference) ** 2))
    if denominator <= 0.:
        raise ValueError("Reference has zero squared norm")
    return float(np.sqrt(np.sum(abs(estimate-reference) ** 2) / denominator))


def reference_masks(reference):
    """Fix both masks from the reference; predicted support cannot hide errors."""
    mask = abs(reference) >= np.exp(-1.) * np.max(abs(reference))
    interior = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), bool))
    if not np.any(mask) or not np.any(interior):
        raise ValueError("Evaluator reference has no valid eroded phase/strain support")
    return mask, interior


def nominal_strain_coefficients(shape):
    """Independent nominal geometry: q in rad/nm, direct basis in nm/index."""
    delta, gamma, step = np.deg2rad([28.488, .0478, .002])
    c = 2. * np.pi / (1.2398419843320026 / 9.)
    column = -c * 55000. / 2.8e9 * np.array([-np.cos(delta), 0., np.sin(delta)])
    row = c * 55000. / 2.8e9 * np.array([
        np.sin(delta)*np.sin(gamma), -np.cos(gamma), np.cos(delta)*np.sin(gamma)])
    rock = c * step * np.array([
        1.-np.cos(delta)*np.cos(gamma), 0., np.sin(delta)*np.cos(gamma)])
    reciprocal = np.column_stack((rock, row, column))
    direct = 2.*np.pi*np.linalg.inv(reciprocal).T @ np.diag(1./np.asarray(shape))
    q = c*np.array([np.cos(gamma)*np.sin(delta), np.sin(gamma),
                    np.cos(delta)*np.cos(gamma)-1.])
    return (q @ np.linalg.inv(direct).T) / np.dot(q, q)


def relative_projected_strain(obj, interior):
    """Central complex ratios, evaluated only on the fixed eroded interior."""
    coordinates = np.nonzero(interior)
    strain = np.zeros(len(coordinates[0]), dtype=np.float64)
    for axis, coefficient in enumerate(nominal_strain_coefficients(obj.shape)):
        plus, minus = list(coordinates), list(coordinates)
        plus[axis] = coordinates[axis]+1
        minus[axis] = coordinates[axis]-1
        ratio = obj[tuple(plus)] * np.conj(obj[tuple(minus)])
        strain += coefficient * np.angle(ratio) / 2.
    return strain-strain.mean()


def phase_strain_errors(aligned, reference):
    mask, interior = reference_masks(reference)
    phase_difference = np.angle(aligned[mask] * np.conj(reference[mask]))
    # Undefined predicted phases must not disappear from the fixed support.
    phase_difference[abs(aligned[mask]) == 0.] = np.pi
    phase = float(np.sqrt(np.mean(phase_difference**2)))
    difference = (relative_projected_strain(aligned, interior) -
                  relative_projected_strain(reference, interior))
    return phase, float(np.sqrt(np.mean(difference**2)))


def measured_consistency(obj, kernel, counts):
    """Amplitude residual on the entire public volume, with no fitted scale."""
    difference = np.asarray(obj.shape)-np.asarray(counts.shape)
    if np.any(difference < 0):
        raise ValueError("Reconstruction grid cannot be smaller than measured grid")
    front = difference//2
    back = difference-front
    peak = np.asarray(np.unravel_index(np.argmax(counts), counts.shape))+front
    shift = np.asarray(obj.shape)//2-peak
    amplitude = np.sqrt(counts.astype(np.float32))
    wave = fft.fftshift(fft.ifftn(fft.ifftshift(obj), norm="forward"))
    predicted = np.sqrt(np.maximum(signal.fftconvolve(abs(wave)**2, kernel, mode="same"), 0.))
    predicted = np.roll(predicted,-shift,axis=(0,1,2))
    predicted = predicted[tuple(slice(int(left),n-int(right) if right else None)
                                for n,left,right in zip(obj.shape,front,back))]
    return float(np.sum((predicted-amplitude)**2) / np.sum(amplitude.astype(np.float64)**2))


def compact_support_quality(obj):
    """Own-field peak gauge and both conjugate-inversion representatives."""
    amplitude=abs(obj);norm=float(np.sum(amplitude**2))
    if norm<=0:return 1.,1.,obj
    aliases=[]
    for current in (obj,np.conj(obj[::-1,::-1,::-1])):
        peak=np.asarray(np.unravel_index(np.argmax(abs(current)),current.shape))
        centered=np.roll(current,np.asarray(current.shape)//2-peak,axis=(0,1,2))
        magnitude=abs(centered);blur=ndimage.gaussian_filter(magnitude,sigma=1.,mode='reflect',truncate=4.)
        support=blur>=.1*blur.max()
        fraction=float(support.mean());outside=float(np.sum(magnitude[~support]**2)/norm)
        aliases.append((outside,fraction,centered))
    return min(aliases,key=lambda item:(item[0],item[1]))[1],min(aliases,key=lambda item:(item[0],item[1]))[0],min(aliases,key=lambda item:(item[0],item[1]))[2]


def score_arrays(prediction,reference,counts):
    measured={}
    for position in range(2):
        obj=prediction['reconstruction'][0,position];kernel=prediction['coherence'][0,position];target=reference['reconstruction'][0,position];prefix=f'p{position+1}_'
        fraction,outside,centered=compact_support_quality(obj)
        _,standard_outside,_=compact_support_quality(target)
        measured[prefix+'support_fraction']=fraction
        measured[prefix+'outside_support_excess']=float(max(0.,outside-standard_outside))
        measured[prefix+'kernel_mass_error']=float(abs(kernel.sum()-1))
        measured[prefix+'kernel_negative_mass']=float(-np.minimum(kernel,0).sum())
        measured[prefix+'raw_amplitude_residual']=measured_consistency(obj,kernel,counts[position])
    if not np.isfinite(list(measured.values())).all():raise ValueError('Nonfinite physical/compactness metric')
    return measured


def strain_diagnostics(prediction):
    diagnostics={}
    for position in range(2):
        obj=prediction['reconstruction'][0,position];amplitude=abs(obj)
        mask=(amplitude>=np.exp(-1)*amplitude.max())&(amplitude>0)
        interior=ndimage.binary_erosion(mask,structure=np.ones((3,3,3),bool))
        if interior.any():
            strain=relative_projected_strain(obj,interior)
            if not np.isfinite(strain).all():raise ValueError('Nonfinite derived physical strain')
            diagnostics[f'p{position+1}']={'unit':'dimensionless','minimum':float(strain.min()),'maximum':float(strain.max()),'mean':float(strain.mean()),'interior_voxels':int(interior.sum())}
        else:diagnostics[f'p{position+1}']={'unit':'dimensionless','interior_voxels':0,'status':'no defined phase interior'}
    return diagnostics


def evaluate(prediction_path):
    result = {"status": "error", "passed": False, "metrics": {},
              "thresholds": {}, "errors": []}
    try:
        prediction = load_arrays(prediction_path)
        reference_path = HERE/"ground_truth.npz"
        if not reference_path.exists():
            reference_path = HERE/"baseline_reference.npz"
        reference = load_arrays(reference_path)
        specification = json.loads((HERE/"metrics.json").read_text(encoding="utf8"))
        thresholds = validate_metric_specification(specification)
        with np.load(HERE.parent/"data/raw_data.npz", allow_pickle=False) as archive:
            counts = archive["photon_counts"]
        if counts.shape != (1, 2, 121, 256, 256) or counts.dtype != np.int32:
            raise ValueError("Evaluator public counts have an invalid shape or dtype")
        validate_reference(reference, counts[0])
        metrics = score_arrays(prediction, reference, counts[0])
        result["derived_strain_diagnostics"] = strain_diagnostics(prediction)
        if set(thresholds) != set(metrics):
            raise ValueError("Threshold names must match every required metric exactly")
        failures = []
        for name, rule in thresholds.items():
            boundary = rule["boundary"]
            comparison = rule["comparison"]
            passed = metrics[name] <= boundary if comparison == "<=" else metrics[name] >= boundary
            if not passed:
                failures.append(f"{name}: {metrics[name]:.17g} does not meet {comparison} {boundary:.17g}")
        result.update(status="fail" if failures else "pass", passed=not failures,
                      metrics=metrics, thresholds=thresholds, errors=failures)
        return result, 1 if failures else 0
    except Exception as exc:
        result["errors"] = [f"{type(exc).__name__}: {exc}"]
        return result, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args(argv)
    result, code = evaluate(Path(arguments.prediction))
    try:
        report = Path(arguments.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    except Exception as exc:
        print(f"Cannot write Judge report: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
