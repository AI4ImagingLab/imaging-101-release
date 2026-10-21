"""Local closure repair, weighted SBAS, spatial ramp and DEM inversion.

Copyright (c) 2013-2026 Zhang Yunjun, Heresh Fattahi and MintPy contributors.
Adapted under GPL-3.0-or-later; attribution and license text: test/validation.md.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import linalg, ndimage

from .physics_model import network_matrices, closure_integers, dem_design
from .preprocessing import REFERENCE, WAVELENGTH, reference_atmosphere


def component_labels(values):
    """Four-neighbour label equal nonzero integers, ordered by first raster pixel."""
    segments = []
    for value in np.unique(values):
        if value == 0:
            continue
        labels, count = ndimage.label(values == value)
        for i in range(1, count + 1):
            indices = np.flatnonzero(labels.ravel() == i)
            segments.append(indices)
    segments.sort(key=lambda indices: indices[0])
    output = np.zeros(values.shape, np.int32)
    for i, indices in enumerate(segments, 1):
        output.ravel()[indices] = i
    return output


def clean_components(values, min_area=2500., erosion_size=5):
    """Preserve MintPy area cap, equal-label connectivity and grey erosion rules."""
    labels = component_labels(values)
    count = int(labels.max())
    if count > 1:
        threshold = min(min_area, values.size * .003)
        area = np.bincount(labels.ravel())
        labels[area[labels] < threshold] = 0
        labels = component_labels(labels)
    if erosion_size > 0:
        eroded = ndimage.minimum_filter(labels, size=erosion_size, mode='reflect').astype(np.uint8)
        remaining = np.unique(eroded)
        labels[~np.isin(labels, remaining)] = 0
        labels = component_labels(labels)
    return labels


def lasso_ambiguities(a, y, alpha=.01, rho=1., tolerance=1e-7, max_iterations=20000):
    """Minimize ||Ax-y||²+alpha||x||1 by explicit shared-factor ADMM.

    The selected MintPy LASSO uses a generic CVXOPT interior-point backend.
    Here the identical convex objective is solved without a domain dependency.
    Its rounded integer solutions, not backend iterates, define closure repair.
    """
    a, y = np.asarray(a, np.float64), np.asarray(y, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    factor = linalg.cho_factor(2 * a.T @ a + rho * np.eye(a.shape[1]))
    rhs = 2 * a.T @ y
    z = np.zeros((a.shape[1], y.shape[1]))
    dual = np.zeros_like(z)
    kkt = np.inf
    for iteration in range(1, max_iterations + 1):
        x = linalg.cho_solve(factor, rhs + rho * (z - dual))
        old = z.copy()
        proposed = x + dual
        z = np.sign(proposed) * np.maximum(np.abs(proposed) - alpha / rho, 0.)
        dual += x - z
        if iteration % 20 == 0:
            gradient = 2 * a.T @ (a @ z - y)
            error = np.where(z != 0, np.abs(gradient + alpha * np.sign(z)), np.maximum(np.abs(gradient) - alpha, 0))
            kkt = float(error.max())
            if kkt < tolerance and np.max(np.abs(x-z)) < tolerance and rho * np.max(np.abs(z-old)) < tolerance:
                return z, {'iterations': iteration, 'kkt_residual': kkt}
    raise RuntimeError(f'LASSO did not converge after {max_iterations} iterations: KKT={kkt}')


def repair_closure(phase, components, c, reference=REFERENCE, num_samples=100,
                   seed=20261018, min_area=2500.):
    """Estimate common-region integer offsets and propagate to local components."""
    common = clean_components(np.all(components > 0, axis=0), min_area=min_area)
    regions = []
    for label in range(1, int(common.max()) + 1):
        coordinates = np.argwhere(common == label)
        rng = np.random.default_rng(seed)
        sample = coordinates[np.sort(rng.choice(len(coordinates), num_samples, replace=False))]
        ambiguity = np.zeros(phase.shape[0])
        diagnostic = {'iterations': 0, 'kkt_residual': 0.}
        if common[reference] != label:
            observations = phase[:, sample[:, 0], sample[:, 1]].copy()
            observations -= phase[:, reference[0], reference[1]][:, None]
            cycles = closure_integers(c.astype(np.float64), observations)
            solution, diagnostic = lasso_ambiguities(-c.astype(np.float64), cycles)
            ambiguity = np.median(np.round(solution), axis=1)
        regions.append({'coords': sample, 'ambiguity': ambiguity, 'area': len(coordinates), **diagnostic})
    regions.sort(key=lambda r: r['area'], reverse=True)
    corrected = np.empty_like(phase)
    for i in range(phase.shape[0]):
        value = phase[i].copy()
        value -= value[reference]
        labels = clean_components(components[i], min_area=min_area)
        # connectComponent.label calls the boundary cleanup a second time.
        labels = clean_components(labels, min_area=0)
        for label in range(1, int(labels.max()) + 1):
            mask = labels == label
            for region in regions:
                sample = region['coords']
                if np.all(mask[sample[:, 0], sample[:, 1]]):
                    value[mask] += 2 * np.pi * region['ambiguity'][i]
                    break
        corrected[i] = value
    return corrected, regions


def estimate_pixel(a, b, phase, delta, weight_sqrt, rcond=1e-5, min_redundancy=1.):
    """Weighted minimum-velocity-norm SVD with original invalid-row semantics."""
    y = np.asarray(phase).reshape(a.shape[0], -1)
    weight = np.asarray(weight_sqrt).reshape(a.shape[0], -1)
    valid = ~np.isnan(y[:, 0])
    aa, bb, y, weight = a[valid], b[valid], y[valid], weight[valid]
    result = np.zeros((a.shape[1] + 1, y.shape[1]), np.float32)
    if np.min(np.sum(aa != 0, axis=0)) < min_redundancy:
        return result, 0., 0
    try:
        velocity = linalg.lstsq(bb * weight, y * weight, cond=rcond)[0]
        residual = y - bb @ velocity
        coherence = np.abs(np.sum(np.exp(1j * residual), axis=0)) / len(y)
        result[1:] = np.cumsum(velocity * np.tile(delta, (1, y.shape[1])), axis=0)
        return result, float(coherence[0]), len(y)
    except linalg.LinAlgError:
        return result, 0., len(y)


def invert_network(corrected, coherence, components, a, b, delta, workers=4):
    """Independent weighted SVDs on the entire spatial stack; threaded by row."""
    dates, height, width = a.shape[1] + 1, corrected.shape[1], corrected.shape[2]
    result = np.zeros((dates, height, width), np.float32)
    quality = np.zeros((height, width), np.float32)
    number = np.zeros((height, width), np.int16)

    def row_solver(row):
        phase = corrected[:, row].copy()
        phase[components[:, row] == 0] = np.nan
        coh = coherence[:, row].copy()
        coh[np.isnan(coh)] = .05
        weight = np.sqrt(np.maximum(coh.astype(np.float64), .05).astype(np.float32))
        output = np.zeros((dates, width), np.float32)
        qq, nn = np.zeros(width, np.float32), np.zeros(width, np.int16)
        for column in range(width):
            ts, q, n = estimate_pixel(a, b, phase[:, column:column+1], delta, weight[:, column:column+1])
            output[:, column] = ts[:, 0] * np.float32(-WAVELENGTH / (4*np.pi))
            qq[column], nn[column] = q, n
        return row, output, qq, nn

    with ThreadPoolExecutor(workers) as pool:
        for row, output, q, n in pool.map(row_solver, range(height)):
            result[:, row], quality[row], number[row] = output, q, n
    return result, quality, number


def unweighted_common_baseline(corrected, mask, b, delta):
    """Actively reconstruct classic uniform-weight SBAS on all-edge-valid pixels."""
    phase = corrected[:, mask]
    velocity = linalg.lstsq(b, phase, cond=1e-5)[0]
    time = np.zeros((b.shape[1]+1, int(mask.sum())), np.float32)
    time[1:] = np.cumsum(velocity * delta, axis=0)
    output = np.zeros((time.shape[0], *mask.shape), np.float32)
    output[:, mask] = time * np.float32(-WAVELENGTH/(4*np.pi))
    residual = phase - b @ velocity
    quality = np.zeros(mask.shape, np.float32)
    quality[mask] = np.abs(np.sum(np.exp(1j*residual), axis=0)) / b.shape[0]
    return output, quality


def deramp_timeseries(data, mask, reference=REFERENCE):
    """Fit the float32 [row,column,1] plane and anchor at the reference pixel."""
    dates, height, width = data.shape
    values = np.moveaxis(data, 0, -1).reshape(-1, dates)
    mean = values.mean(axis=-1)
    selected = mask.ravel() & np.isfinite(mean) & (mean != 0)
    yy, xx = np.mgrid[:height, :width]
    g = np.column_stack((yy.ravel(), xx.ravel(), np.ones(height * width))).astype(np.float32)
    parameters = np.linalg.pinv(g[selected], rcond=1e-15) @ values[selected]
    ramp = (g @ parameters).astype(data.dtype)
    ramp -= ramp[reference[0] * width + reference[1]]
    ramp[values == 0] = 0
    corrected = values - ramp
    return np.moveaxis(corrected.reshape(height, width, dates), -1, 0), np.moveaxis(ramp.reshape(height, width, dates), -1, 0)


def correct_dem(data, design, rcond=1e-8):
    """Jointly estimate DEM residual and polynomial/eruption coefficients."""
    values = data.reshape(data.shape[0], -1)
    parameters = linalg.lstsq(design, values, cond=rcond)[0]
    corrected = values - design[:, :1] @ parameters[:1]
    residual = values - design @ parameters
    shape = data.shape[1:]
    return (corrected.reshape(data.shape).astype(np.float32), parameters[0].reshape(shape).astype(np.float32),
            parameters[1:].reshape(-1, *shape).astype(np.float32), residual.reshape(data.shape).astype(np.float32))


def reconstruct(data, workers=4):
    """Run the complete selected method; return required quantities and diagnostics."""
    a, b, c, delta = network_matrices(data['dates'], data['pairs'])
    corrected, regions = repair_closure(data['phase'], data['components'], c)
    network, quality, number = invert_network(corrected, data['coherence'], data['components'], a, b, delta, workers)
    atmosphere = reference_atmosphere(data['atmospheric_delay'])
    tropo = network - atmosphere
    fit_mask = (number == len(a)) & (quality >= .7)
    deramped, ramp = deramp_timeseries(tropo, fit_mask)
    design = dem_design(data['baseline'], data['incidence'], data['dates'])
    final, dem, coefficients, residual = correct_dem(deramped, design)
    output = dict(reconstruction=final, dem_error=dem, deformation_parameters=coefficients,
                  temporal_coherence=quality, network_timeseries=network)
    diagnostics = dict(corrected_phase=corrected, regions=regions, num_observations=number,
                       atmosphere=atmosphere, troposphere_corrected=tropo, deramped=deramped,
                       spatial_ramp=ramp, residual=residual, fit_mask=fit_mask, design=design)
    return output, diagnostics
