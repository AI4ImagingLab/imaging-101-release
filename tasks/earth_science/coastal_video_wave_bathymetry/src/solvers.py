"""Local UBathy inverse updates and complete wave-to-bed reconstruction.
Derived from the AGPL-3.0 UBathy implementation by Gonzalo Simarro and Daniel Calvete.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from .preprocessing import PARAMETERS, G, recordings, window_specs
from .physics_model import (
    analytic_signal, dmd, dispersion_wavenumber, gamma_from_bed,
    phase_difference, ransac_trial_count,
)

def _parameters(parameters):
    return {**PARAMETERS, **({} if parameters is None else parameters)}

def fit_phase_plane(x, y, phase, rng, parameters=None):
    """Native sequential choice, adaptive consensus and normal-equation refit."""
    p = _parameters(parameters)
    x, y, phase = map(np.asarray, (x, y, phase))
    if x.ndim != 1 or x.shape != y.shape or x.shape != phase.shape or len(x) < 3:
        raise ValueError("Phase-plane observations must have matching length >= 3")
    a = np.column_stack((x, y, np.ones(len(x))))
    if np.linalg.det(a.T @ a) == 0:
        raise ValueError("Collinear phase samples cannot identify a plane")
    best = np.empty(0, dtype=int)
    required = np.inf
    trials = draws = 0
    while trials < required:
        proposal = rng.choice(range(len(x)), size=3, replace=False)
        sampled, target = a[proposal], phase[proposal]
        draws += 1
        if draws > 20000:
            raise ValueError("Unable to sample nonsingular phase planes")
        try:
            coefficients = np.linalg.solve(sampled.T @ sampled, sampled.T @ target)
            errors = np.abs(a @ coefficients - phase)
        except np.linalg.LinAlgError:
            continue
        selected = np.flatnonzero(errors <= p["phase_cutoff"])
        if len(selected) > len(best):
            best = selected
            outlier = 1 - len(best) / len(x) + p["ransac_margin"] * len(best) / len(x)
            required = min(p["ransac_max_trials"], ransac_trial_count(
                outlier, p["ransac_probability"], 3))
        if len(best) == len(x):
            break
        trials += 1
    selected = a[best]
    coefficients = np.linalg.solve(selected.T @ selected, selected.T @ phase[best])
    return coefficients, best, {
        "trials": trials, "draws": draws, "consensus_count": len(best),
        "samples": len(x),
    }

def extract_modes(data, parameters=None):
    """Compute all native windows and selected wave modes from both raw count records."""
    p = _parameters(parameters)
    result = []
    for record in recordings(data):
        record = dict(record)
        modes = {}
        spectra = []
        analytic_cache = {}
        for spec in window_specs(record, p):
            start, nh = spec["start"], spec["nh"]
            if start not in analytic_cache:
                extended = record["intensity"][:, start:start + spec["nmax"] + 2 * nh]
                analytic_cache[start] = analytic_signal(extended)
            values = analytic_cache[start][:, nh:-nh][:, :spec["columns"]]
            modal = dmd(values, rank=p["rank"], dt=record["dt"])
            spectra.append({
                "start_seconds": spec["start_time"], "window_seconds": spec["seconds"],
                "singular_values": modal["singular_values_full"],
                "periods": modal["periods"],
            })
            for j, period in enumerate(modal["periods"]):
                if not p["min_period"] <= period <= p["max_period"]:
                    continue
                key = (f"t{spec['start_time']:08.2f}_wt{spec['seconds']:08.2f}"
                       f"_T{period:08.2f}")
                modes[key] = {
                    "key": key, "record": record["label"],
                    "start_seconds": spec["start_time"],
                    "window_seconds": spec["seconds"],
                    "period": float(period), "omega": float(2 * np.pi / period),
                    "phases": modal["phases"][j].copy(),
                    "amplitudes": modal["amplitudes"][j].copy(),
                }
        record["modes"] = [modes[key] for key in sorted(modes)]
        record["spectra"] = spectra
        result.append(record)
    return result

def estimate_wave_fields(records, seed=20261026, parameters=None, verbose=False):
    """Estimate every selected mode at every native wave-grid site and all three radii."""
    p = _parameters(parameters)
    rng = np.random.RandomState(seed)
    fields = []
    for record in records:
        xy, kxy = record["mode_xy"], record["wave_xy"]
        mode_tree = cKDTree(xy)
        wave_tree = cKDTree(kxy)
        for mode in record["modes"]:
            started = time.perf_counter()
            h_aux = np.array([
                p["min_depth"] + (j + 1) / p["neighborhoods"] *
                (p["max_depth"] - p["min_depth"])
                for j in range(p["neighborhoods"])
            ])[::-1]
            radii = p["radius_k"] * 2 * np.pi / dispersion_wavenumber(mode["omega"], h_aux)
            kx, ky = [np.full((len(kxy), len(radii)), np.nan) for _ in range(2)]
            candidates = mode_tree.query_ball_point(kxy, float(radii[0]) * (1 + 1e-12),
                                                   return_sorted=True)
            for pos, neighbors in enumerate(candidates):
                if len(neighbors) < 6:
                    continue
                neighbors = np.asarray(neighbors, dtype=int)
                distance = np.hypot(xy[neighbors, 0] - kxy[pos, 0],
                                    xy[neighbors, 1] - kxy[pos, 1])
                inside = distance <= radii[0]
                neighbors, distance = neighbors[inside], distance[inside]
                if len(neighbors) < 6:
                    continue
                nearest = neighbors[np.argmin(distance)]
                origin = xy[nearest]
                reference_phase = mode["phases"][nearest]
                for j, radius in enumerate(radii):
                    selected = neighbors[distance <= radius]
                    if len(selected) < 6:
                        continue
                    centroid_offset = np.hypot(np.mean(xy[selected, 0]) - kxy[pos, 0],
                                               np.mean(xy[selected, 1]) - kxy[pos, 1])
                    if centroid_offset > radius / 4:
                        continue
                    dx = xy[selected, 0] - origin[0]
                    dy = xy[selected, 1] - origin[1]
                    phase = phase_difference(mode["phases"][selected], reference_phase)
                    try:
                        coeff, inliers, _ = fit_phase_plane(dx, dy, phase, rng, p)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if len(inliers) >= 6:
                        kx[pos, j], ky[pos, j] = coeff[0], coeff[1]
            ks = np.hypot(kx, ky)
            reject = (ks < 1e-6) | (1.2 * ks < mode["omega"] ** 2 / G)
            ks[reject] = np.nan
            gamma = mode["omega"] ** 2 / (G * ks)
            mean, std = [np.full_like(ks, np.nan) for _ in range(2)]
            for j in range(len(radii)):
                good = np.flatnonzero(np.isfinite(ks[:, j]))
                for pos in good:
                    radius = max(.5 * 2 * np.pi / ks[pos, j], 2.1 * p["delta_k"])
                    near = np.asarray(wave_tree.query_ball_point(
                        kxy[pos], radius * (1 + 1e-12)), dtype=int)
                    near.sort()
                    distance = np.hypot(kxy[near, 0] - kxy[pos, 0],
                                        kxy[near, 1] - kxy[pos, 1])
                    near = near[(distance <= radius) & np.isfinite(ks[near, j])]
                    if len(near) >= 3:
                        mean[pos, j] = np.mean(gamma[near, j])
                        std[pos, j] = np.std(gamma[near, j])
            fields.append({
                **{k: mode[k] for k in (
                    "record", "key", "start_seconds", "window_seconds", "period", "omega")},
                "radii": radii, "ks": ks, "mean_gamma": mean, "std_gamma": std,
                "wave_xy": kxy, "water_level": record["water_level"],
            })
            if verbose:
                print(mode["record"], mode["key"], "wave_seconds",
                      round(time.perf_counter() - started, 3), flush=True)
    return fields

def aggregate_wave_observations(fields, parameters=None):
    """Native observation-only gamma qualification and planview/oblique aggregation."""
    p = _parameters(parameters)
    parts = {k: [] for k in ("xy", "gamma", "mean_gamma", "omega", "water_level")}
    # This is the native videos4dates order, distinct from processing/file order.
    for label in ("planview", "oblique"):
        for field in sorted((f for f in fields if f["record"] == label), key=lambda f: f["key"]):
            omega = field["omega"]
            for j in range(field["ks"].shape[1]):
                ks = field["ks"][:, j]
                mean = field["mean_gamma"][:, j]
                std = field["std_gamma"][:, j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    gamma = omega ** 2 / (G * ks)
                    difference = np.abs(gamma - mean)
                good = (np.isfinite(std) & np.isfinite(difference) &
                        (std <= p["gamma_cutoff"]) & (difference <= p["gamma_cutoff"]))
                count = int(good.sum())
                if not count:
                    continue
                parts["xy"].append(field["wave_xy"][good])
                parts["gamma"].append(gamma[good])
                parts["mean_gamma"].append(mean[good])
                parts["omega"].append(np.full(count, omega))
                parts["water_level"].append(np.full(count, field["water_level"]))
    return {
        key: np.concatenate(values) if values else np.empty((0, 2) if key == "xy" else 0)
        for key, values in parts.items()
    }

def _bed_objective(value, gamma, omega, water):
    """The original dimensionless gamma RMS, with no averaging convention changed."""
    predicted = gamma_from_bed(omega, water, float(np.asarray(value)[0]))
    return float(np.sqrt(np.mean((predicted - gamma) ** 2)))

def fit_bed_point(gamma, omega, water, initial):
    """Native unconstrained BFGS polishing of one already selected consensus set."""
    result = minimize(_bed_objective, np.array([initial]),
                      args=(gamma, omega, water))
    # Preserve the original returned x even when BFGS reports precision loss.
    return float(result.x[0]), {
        "optimizer_success": bool(result.success), "evaluations": int(result.nfev),
        "gamma_rmse": _bed_objective(result.x, gamma, omega, water),
    }

def spatial_uncertainty(xy, raw_bed, radii):
    """Native population spatial SD, deliberately propagating any unsupported neighbor."""
    tree = cKDTree(xy)
    uncertainty = np.full(len(xy), np.nan)
    for pos in np.flatnonzero(np.isfinite(raw_bed) & np.isfinite(radii)):
        radius = radii[pos]
        near = np.asarray(tree.query_ball_point(xy[pos], radius * (1 + 1e-12)), dtype=int)
        near.sort()
        distance = np.hypot(xy[near, 0] - xy[pos, 0], xy[near, 1] - xy[pos, 1])
        near = near[distance <= radius]
        if len(near) >= 3:
            uncertainty[pos] = np.std(raw_bed[near])
    return uncertainty

def fit_bed(reconstruction_xy, observations, parameters=None):
    """Full-domain consensus-grid/BFGS inversion and observation-support validity."""
    p = _parameters(parameters)
    xy = np.asarray(reconstruction_xy)
    n = len(xy)
    raw_bed, radii, gamma_rmse = [np.full(n, np.nan) for _ in range(3)]
    consensus_counts = np.zeros(n, dtype=np.int64)
    support_counts = np.zeros(n, dtype=np.int64)
    optimizer_success = np.zeros(n, dtype=bool)
    source_xy = observations["xy"]
    if len(source_xy):
        tree = cKDTree(source_xy)
        distances, _ = tree.query(xy)
        for pos, nearest_distance in enumerate(distances):
            close_radius = 1.1 * nearest_distance + .001
            close = np.asarray(tree.query_ball_point(
                xy[pos], close_radius * (1 + 1e-12)), dtype=int)
            close.sort()
            dclose = np.hypot(source_xy[close, 0] - xy[pos, 0],
                              source_xy[close, 1] - xy[pos, 1])
            close = close[dclose <= close_radius]
            wavelength = np.mean(2 * np.pi * G * observations["mean_gamma"][close] /
                                 observations["omega"][close] ** 2)
            radius = max(p["radius_b"] * wavelength, 2.1 * p["delta_k"])
            near = np.asarray(tree.query_ball_point(
                xy[pos], radius * (1 + 1e-12)), dtype=int)
            near.sort()
            distance = np.hypot(source_xy[near, 0] - xy[pos, 0],
                                source_xy[near, 1] - xy[pos, 1])
            near = near[distance <= radius]
            support_counts[pos] = len(near)
            if len(near) < 10:
                continue
            gamma, omega, water = [observations[k][near] for k in
                                   ("gamma", "omega", "water_level")]
            heights = np.arange(water.min() - p["max_depth"],
                                water.max() - p["min_depth"], .05)
            if not len(heights):
                continue
            depth = water[None, :] - heights[:, None]
            predicted = omega[None, :] ** 2 / (
                G * dispersion_wavenumber(omega[None, :],
                                           np.clip(depth, p["min_depth"], np.inf)))
            predicted[depth <= p["min_depth"]] = np.inf
            inliers = np.abs(predicted - gamma[None, :]) < p["gamma_cutoff"]
            counts = inliers.sum(axis=1)
            best = int(np.argmax(counts))  # Native first maximum in ascending bed grid.
            consensus_counts[pos] = int(counts[best])
            if counts[best] < 10:
                continue
            selected = inliers[best]
            estimate, info = fit_bed_point(gamma[selected], omega[selected],
                                           water[selected], heights[best])
            raw_bed[pos] = estimate
            radii[pos] = radius
            gamma_rmse[pos] = info["gamma_rmse"]
            optimizer_success[pos] = info["optimizer_success"]
    uncertainty = spatial_uncertainty(xy, raw_bed, radii)
    valid = np.isfinite(raw_bed) & np.isfinite(uncertainty)
    submission = {
        "reconstruction": np.where(valid, raw_bed, 0.0)[None],
        "validity": valid[None],
    }
    diagnostics = {
        "raw_bed": raw_bed, "uncertainty": uncertainty, "radii": radii,
        "consensus_counts": consensus_counts, "support_counts": support_counts,
        "gamma_rmse": gamma_rmse, "optimizer_success": optimizer_success,
    }
    return submission, diagnostics

def reconstruct(data, seed=20261026, return_diagnostics=False, parameters=None):
    """Active complete two-record, twenty-mode inverse; never reads a reference/cache."""
    timings = {}
    start = time.perf_counter()
    records = extract_modes(data, parameters)
    timings["modes_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    fields = estimate_wave_fields(records, seed=seed, parameters=parameters, verbose=True)
    timings["wave_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    observations = aggregate_wave_observations(fields, parameters)
    timings["aggregation_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    submission, bed = fit_bed(data["reconstruction_xy"], observations, parameters)
    timings["bed_seconds"] = time.perf_counter() - start
    diagnostics = {
        "records": records, "wave_fields": fields, "observations": observations,
        "bed": bed, "timings": timings,
    }
    return (submission, diagnostics) if return_diagnostics else submission
