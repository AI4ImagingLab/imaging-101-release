"""Local linear algebra and ADJUST constrained spectral optimization.

ADJUST method: Copyright (c) 2021 mzeegers, MIT.
SPG: Mark Schmidt's MinConf implementation; upstream notices and versions are
preserved in test/validation.md. No upstream module is imported at runtime.
Scaled Frobenius accumulation: GNU Octave 8.4.0, Copyright (C) 2008-2023
The Octave Project Developers, GPL-3.0-or-later; notice in test/validation.md.
"""
from __future__ import annotations

import random
import math
import time
import numpy as np
from .linear_algebra import current_backend


def cast_sparse_preserving_order(matrix, dtype):
    """Change precision without SciPy astype's duplicate/sort side effect."""
    from scipy.sparse import csr_matrix
    csr = matrix.tocsr()
    return csr_matrix((csr.data.astype(dtype), csr.indices.copy(), csr.indptr.copy()),
                      shape=csr.shape)


class LinearMap:
    """Apply a fixed explicit sparse operator with NumPy or CUDA primitives."""

    def __init__(self, matrix, backend='cpu'):
        if backend not in ('cpu', 'cpu_sparse', 'cuda', 'cuda_sparse', 'cuda_ordered'):
            raise ValueError('backend must be cpu, cpu_sparse, cuda, cuda_sparse, or cuda_ordered')
        self.backend = backend
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        if backend == 'cuda_ordered':
            from .deterministic_sparse import OrderedCsr
            self.matrix = matrix.tocsr()
            self.transposed = matrix.T.tocsr()
            self.ordered_forward = OrderedCsr(self.matrix)
            self.ordered_transpose = OrderedCsr(self.transposed)
        elif backend in ('cuda', 'cuda_sparse'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            self.device = 'cuda'
            self.tdtype = torch.float64 if matrix.dtype == np.float64 else torch.float32
            csr = matrix.tocsr()
            self.matrix = torch.sparse_csr_tensor(
                torch.from_numpy(csr.indptr.astype(np.int64)),
                torch.from_numpy(csr.indices.astype(np.int64)),
                torch.from_numpy(csr.data), size=csr.shape,
                device=self.device, dtype=self.tdtype)
            transpose = matrix.T.tocsr()
            self.transposed = torch.sparse_csr_tensor(
                torch.from_numpy(transpose.indptr.astype(np.int64)),
                torch.from_numpy(transpose.indices.astype(np.int64)),
                torch.from_numpy(transpose.data), size=transpose.shape,
                device=self.device, dtype=self.tdtype)
        else:
            self.matrix = matrix.tocsr()
            self.transposed = matrix.T.tocsr()

    def array(self, value):
        if self.backend == 'cuda':
            import torch
            return torch.as_tensor(value, device=self.device, dtype=self.tdtype)
        if self.backend in ('cpu_sparse', 'cuda_sparse', 'cuda_ordered'):
            return np.asfortranarray(value, dtype=self.dtype)
        return np.asarray(value, dtype=self.dtype)

    def zeros(self, shape):
        if self.backend == 'cuda':
            import torch
            return torch.zeros(shape, device=self.device, dtype=self.tdtype)
        return np.zeros(shape, dtype=self.dtype,
                        order='F' if self.backend in ('cpu_sparse', 'cuda_sparse', 'cuda_ordered') else 'C')

    def _host_sparse_product(self, matrix, value):
        import torch
        # Match the source relay's contiguous input and column-major host return.
        dense = torch.as_tensor(np.ascontiguousarray(value), device=self.device, dtype=self.tdtype)
        result = torch.sparse.mm(matrix, dense).cpu().numpy()
        return np.asfortranarray(result)

    def multiply(self, value):
        if self.backend == 'cuda_ordered':
            return self.ordered_forward.multiply(value)
        if self.backend == 'cpu_sparse':
            return np.asfortranarray(self.matrix @ np.ascontiguousarray(value))
        if self.backend == 'cuda_sparse':
            return self._host_sparse_product(self.matrix, value)
        if self.backend == 'cuda':
            import torch
            return torch.sparse.mm(self.matrix, value)
        return self.matrix @ value

    def transpose(self, value):
        if self.backend == 'cuda_ordered':
            return self.ordered_transpose.multiply(value)
        if self.backend == 'cpu_sparse':
            return np.asfortranarray(self.transposed @ np.ascontiguousarray(value))
        if self.backend == 'cuda_sparse':
            return self._host_sparse_product(self.transposed, value)
        if self.backend == 'cuda':
            import torch
            return torch.sparse.mm(self.transposed, value)
        return self.transposed @ value

    def numpy(self, value):
        return value.detach().cpu().numpy() if self.backend == 'cuda' else np.asarray(value)


def initial_state(pixels: int, materials: int, atoms: int, seed: int = 10):
    """Octave twister initialization, preserving column-major random draw order."""
    rng = random.Random(seed)
    maps = np.fromiter((rng.random() for _ in range(pixels * materials)),
                       dtype=np.float64).reshape(pixels, materials, order='F')
    maps /= np.sum(maps, axis=1, keepdims=True)
    coefficients = np.fromiter((rng.random() for _ in range(materials * atoms)),
                               dtype=np.float64).reshape(materials, atoms, order='F')
    return maps, coefficients


def _torch(value):
    return type(value).__module__.startswith('torch')


def _host(value):
    return value.detach().cpu().numpy() if _torch(value) else np.asarray(value)


def _sum(value):
    values = _host(value).ravel(order='F')
    return float(np.cumsum(values)[-1]) if values.size else 0.0


def _dot(left, right):
    backend = current_backend()
    if backend is not None:
        return backend.dot(_host(left), _host(right))
    return float(np.dot(_host(left).ravel(order='F'), _host(right).ravel(order='F')))


def _matmul(left, right):
    backend = current_backend()
    if backend is not None and not _torch(left):
        return backend.matmul(np.asarray(left), np.asarray(right))
    return left @ right


def frobenius_norm(value):
    """Explicit source-compatible serial scaled sum in column-major order.

    Between record maxima the scale is constant, so NumPy cumsum evaluates
    the same ordered additions without a Python loop over every array entry.
    """
    values = np.abs(np.asarray(_host(value), dtype=np.float64).ravel(order='F'))
    if not len(values):
        return 0.0
    if not np.all(np.isfinite(values)):
        return float(np.linalg.norm(values))
    running = np.maximum.accumulate(values)
    records = np.flatnonzero(np.r_[values[0] > 0, values[1:] > running[:-1]])
    del running
    scale, total = 0.0, 1.0
    for number, start in enumerate(records):
        new_scale = float(values[start])
        ratio = scale / new_scale
        total *= ratio * ratio
        total += 1.0
        scale = new_scale
        stop = records[number + 1] if number + 1 < len(records) else len(values)
        if stop > start + 1:
            terms = np.empty(stop - start, dtype=np.float64)
            terms[0] = total
            np.divide(values[start + 1:stop], scale, out=terms[1:])
            np.square(terms[1:], out=terms[1:])
            np.cumsum(terms, out=terms)
            total = float(terms[-1])
    return scale * math.sqrt(total)


def _norm(value):
    return frobenius_norm(value)


def _scalar_power(base, exponent):
    backend = current_backend()
    if backend is not None:
        return backend.power(base, exponent)
    return math.pow(float(base), float(exponent))


def _maxabs(value):
    return float(value.abs().max().item()) if _torch(value) else float(np.max(np.abs(value)))


def project_simplex_columns(values: np.ndarray, mass: float = 1.0,
                            equality: bool = False) -> np.ndarray:
    """Original per-column equality/sub-simplex projection, independently stated."""
    x = np.asarray(values, dtype=np.float64)
    if (np.any(x < 0) or np.any(np.sum(x, axis=0) > mass)
            or (equality and np.linalg.norm(x.sum(axis=0) - mass) > 1e-6 / abs(mass))):
        ordered = np.sort(x, axis=0)[::-1]
        if mass < np.spacing(ordered[0]).max():
            raise ValueError('simplex mass too small for input scale')
        reciprocal = 1.0 / np.arange(1, len(x) + 1, dtype=np.float64)
        average = reciprocal[:, None] * (np.cumsum(ordered, axis=0) - mass)
        count = np.sum(ordered > average, axis=0)
        tau = average[count - 1, np.arange(x.shape[1])]
        if not equality:
            tau = np.maximum(tau, 0)
        x = np.maximum(x - tau[None, :], 0)
    return x.copy()


def project_material_maps(values):
    """Project every pixel's material fractions onto nonnegative sum<=1."""
    if not _torch(values):
        return project_simplex_columns(values.T).T
    import torch
    ordered = torch.sort(values, dim=1, descending=True).values
    reciprocal = 1 / torch.arange(
        1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    avg = reciprocal[None, :] * (torch.cumsum(ordered, dim=1) - 1)
    count = (ordered > avg).sum(dim=1)
    tau = torch.clamp(avg.gather(1, (count - 1)[:, None]), min=0)
    return torch.clamp(values - tau, min=0)


def project_dictionary_weights(values):
    """Upstream alternating row-equality/column-sub-simplex projection.

    Correct the documented upstream sum>=1 skip bug: every trial must satisfy
    the published nonnegativity/equality constraints. Preserve its alternating
    projections, 1e-6 discrepancy tolerance and 1e4 maximum passes.
    """
    is_tensor = _torch(values)
    x = values.detach().cpu().numpy().copy() if is_tensor else np.asarray(values).copy()
    for _ in range(10000):
        row_projected = project_simplex_columns(x.T, equality=True).T
        x = project_simplex_columns(row_projected)
        if np.linalg.norm(row_projected - x) < 1e-6:
            break
    if is_tensor:
        import torch
        return torch.as_tensor(x, device=values.device, dtype=values.dtype)
    return x


def spectral_objective(weights, data, projected_maps, dictionary, norm_data):
    """Return 0.5||WA R T-Y||_F^2/||Y||_F and its R gradient."""
    residual = _matmul(projected_maps, _matmul(weights, dictionary)) - data
    return 0.5 * _scalar_power(_norm(residual), 2.) / norm_data, (
        _matmul(projected_maps.T, _matmul(residual, dictionary.T))) / norm_data


def spatial_objective(maps, spectra, data, operator: LinearMap, norm_data):
    """Return the matching objective and exact W-transpose spatial gradient."""
    residual = _matmul(operator.multiply(maps), spectra) - data
    return 0.5 * _scalar_power(_norm(residual), 2.) / norm_data, operator.transpose(
        _matmul(residual, spectra.T)) / norm_data


def _cubic_step(f, gtd, step, f_new, derivative_new):
    d1 = gtd + derivative_new - 3 * (f - f_new) / (-step)
    discriminant = _scalar_power(d1, 2.) - gtd * derivative_new
    if discriminant < 0:
        return step * 0.5
    d2 = np.sqrt(discriminant)
    denominator = derivative_new - gtd + 2 * d2
    candidate = step - step * ((derivative_new + d2 - d1) / denominator) if denominator else step * .5
    return float(np.clip(candidate, 0, step))


def spectral_projected_gradient(objective, values, projection, max_evaluations=2,
                                memory=2, progress_tolerance=1e-12,
                                sufficient_decrease=1e-12):
    """Explicit MinConf-compatible BB1, curvilinear nonmonotone SPG updates."""
    x = projection(values)
    f, g = objective(x)
    evaluations, iteration = 1, 1
    previous_values = []
    x_old = g_old = None
    while evaluations <= max_evaluations:
        alpha = 1.0
        if iteration > 1:
            s, y = x - x_old, g - g_old
            denominator = _dot(s, y)
            alpha = _dot(s, s) / denominator if denominator else np.inf
            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1.0
        direction = -alpha * g
        f_old, x_old, g_old = f, x, g
        derivative = _dot(g, direction)
        if derivative > -progress_tolerance:
            break
        step = min(1.0, 1.0 / _sum(abs(g))) if iteration == 1 else 1.0
        previous_values.append(f)
        previous_values = previous_values[-memory:]
        reference = max(previous_values)
        x_new = projection(x + step * direction)
        f_new, g_new = objective(x_new)
        evaluations += 1
        while (f_new > reference + _dot(sufficient_decrease * g, x_new - x)
               or not np.isfinite(f_new)):
            old_step = step
            if not np.isfinite(f_new):
                step *= .5
            else:
                step = _cubic_step(f, derivative, step, f_new, _dot(g_new, direction))
            step = float(np.clip(step, old_step * 1e-3, old_step * .6))
            if _maxabs(step * direction) < progress_tolerance or step == 0:
                step, f_new, g_new = 0.0, f, g
                break
            x_new = projection(x + step * direction)
            f_new, g_new = objective(x_new)
            evaluations += 1
        x, f, g = x_new, f_new, g_new
        if (_maxabs(step * direction) < progress_tolerance
                or abs(f - f_old) < progress_tolerance
                or evaluations > max_evaluations):
            break
        iteration += 1
    return x


def adjust(data: np.ndarray, matrix, dictionary: np.ndarray, materials: int = 4,
           iterations: int = 1000, inner_evaluations: int = 2, rho: float = .01,
           seed: int = 10, backend: str = 'cpu', verbose: bool = True):
    """Full dictionary-constrained joint reconstruction, dynamic residual lift.

    The norm denominator, inner evaluation counter, update ordering, best
    iterate and stopping tests follow the frozen original ADJUST source.
    """
    operator = LinearMap(cast_sparse_preserving_order(matrix, np.float64), backend)
    y, t = operator.array(data), operator.array(dictionary)
    maps, weights = initial_state(matrix.shape[1], materials, len(dictionary), seed)
    maps, weights = operator.array(maps), operator.array(weights)
    q = operator.zeros(data.shape)
    norm_data = _norm(y)
    if not np.isfinite(norm_data) or norm_data == 0:
        raise ValueError('observations must have a finite nonzero norm')
    best_residual = np.inf
    history = []
    feasibility = []
    start = time.perf_counter()
    for iteration in range(iterations):
        old_maps, old_weights = maps, weights
        lifted_data = y + rho * q
        projected = operator.multiply(maps)
        weights = spectral_projected_gradient(
            lambda r: spectral_objective(r, lifted_data, projected, t, norm_data),
            weights, project_dictionary_weights, inner_evaluations, inner_evaluations)
        spectra = _matmul(weights, t)
        maps = spectral_projected_gradient(
            lambda a: spatial_objective(a, spectra, lifted_data, operator, norm_data),
            maps, project_material_maps, inner_evaluations, inner_evaluations)
        residual = y - _matmul(operator.multiply(maps), spectra)
        q = q + residual
        relative = _norm(residual) / norm_data
        progress = _norm(weights - old_weights) + _norm(maps - old_maps)
        residual_progress = abs(relative - history[-1][0]) if history else 1.0
        history.append([relative, residual_progress, progress, rho])
        if _torch(maps):
            feasibility.append([max(0., -float(maps.min().item())),
                                max(0., float(maps.sum(dim=1).max().item()) - 1),
                                max(0., -float(weights.min().item())),
                                float((weights.sum(dim=1) - 1).abs().max().item()),
                                max(0., float(weights.sum(dim=0).max().item()) - 1)])
        else:
            feasibility.append([max(0., -float(maps.min())), max(0., float(maps.sum(axis=1).max()) - 1),
                                max(0., -float(weights.min())), float(abs(weights.sum(axis=1) - 1).max()),
                                max(0., float(weights.sum(axis=0).max()) - 1)])
        if relative <= best_residual:
            best_residual = relative
            best_maps = operator.numpy(maps).copy()
            best_weights = operator.numpy(weights).copy()
            best_spectra = operator.numpy(spectra).copy()
        if verbose and (iteration == 0 or (iteration + 1) % 25 == 0):
            print(f'ADJUST {iteration+1}/{iterations}: residual={relative:.9g}, '
                  f'rho={rho:.3g}, elapsed={time.perf_counter()-start:.1f}s', flush=True)
        if relative < 1e-9 or progress < 1e-9 or residual_progress < 1e-9:
            break
        if rho > .05 and residual_progress > .001:
            rho = .01
        if residual_progress < 1e-5:
            rho *= 10
    return {'maps': best_maps, 'spectra': best_spectra, 'weights': best_weights,
            'history': np.asarray(history), 'feasibility_history': np.asarray(feasibility),
            'runtime_seconds': time.perf_counter() - start}


def _nonnegative(values):
    return values.clamp(min=0) if _torch(values) else np.maximum(values, 0)


def _fractions_box(values):
    return values.clamp(0, 1) if _torch(values) else np.clip(values, 0, 1)


def classical_joint(data, matrix, materials=4, iterations=500, seed=10,
                    backend='cpu', verbose=True):
    """Original cJoint: unconstrained spectra, [0,1] maps, residual lifting."""
    op = LinearMap(cast_sparse_preserving_order(matrix, np.float64), backend)
    y = op.array(data)
    # cJoint starts with unnormalized random maps, then random spectral factors.
    rng = random.Random(seed)
    maps = op.array(np.fromiter((rng.random() for _ in range(matrix.shape[1] * materials)),
                                dtype=np.float64).reshape(matrix.shape[1], materials, order='F'))
    spectra = op.array(np.fromiter((rng.random() for _ in range(materials * data.shape[1])),
                                   dtype=np.float64).reshape(materials, data.shape[1], order='F'))
    q = op.zeros(data.shape)
    norm_data, best = _norm(y), np.inf
    history = []
    start = time.perf_counter()
    for it in range(iterations):
        old_maps, old_spectra = maps, spectra
        lifted = y + .01 * q
        projected = op.multiply(maps)

        def objective(factors):
            residual = _matmul(projected, factors) - lifted
            return .5 * _scalar_power(_norm(residual), 2.) / norm_data, _matmul(projected.T, residual) / norm_data

        spectra = spectral_projected_gradient(objective, spectra, _nonnegative,
                                               max_evaluations=3, memory=3,
                                               progress_tolerance=1e-9, sufficient_decrease=1e-9)
        maps = spectral_projected_gradient(
            lambda a: spatial_objective(a, spectra, lifted, op, norm_data),
            maps, _fractions_box, max_evaluations=3, memory=3,
            progress_tolerance=1e-9, sufficient_decrease=1e-9)
        residual = y - _matmul(op.multiply(maps), spectra)
        q += residual
        relative = _norm(residual) / norm_data
        progress = _norm(spectra - old_spectra) + _norm(maps - old_maps)
        history.append([relative, progress])
        if relative < best:
            best = relative
            best_maps, best_spectra = op.numpy(maps).copy(), op.numpy(spectra).copy()
        if verbose and (it == 0 or (it + 1) % 100 == 0):
            print(f'cJoint {it+1}/{iterations}: residual={relative:.9g}', flush=True)
        if relative < 1e-6 or progress < 1e-6:
            break
    return {'maps': best_maps, 'spectra': best_spectra, 'history': np.asarray(history),
            'runtime_seconds': time.perf_counter() - start}


def nonnegative_least_squares_small(gram, right):
    """Exact support enumeration for k<=4 NNLS with multiple right-hand sides."""
    if gram.shape[0] > 4:
        raise ValueError('support enumeration is restricted to four materials')
    k = gram.shape[0]
    best = np.zeros_like(right, dtype=np.float64)
    objective = np.zeros(right.shape[1])
    for bits in range(1, 1 << k):
        active = np.array([i for i in range(k) if bits & (1 << i)])
        block = gram[np.ix_(active, active)]
        candidate = np.linalg.lstsq(block, right[active], rcond=None)[0]
        legal = np.all(candidate >= 0, axis=0)
        value = .5 * np.sum(candidate * (block @ candidate), axis=0) - np.sum(candidate * right[active], axis=0)
        improve = legal & (value < objective)
        best[:, improve] = 0
        best[np.ix_(active, np.flatnonzero(improve))] = candidate[:, improve]
        objective[improve] = value[improve]
    return best


def factor_nonnegative(data, materials=4, replicates=10, multiplicative_iterations=100,
                       als_iterations=30, seed=10):
    """Explicit sequential-comparison NMF, independent of MATLAB's closed nnmf.

    Ten multiplicative starts are followed by exact small-NNLS alternating
    minimization of the best start. This is a disclosed comparison port, not
    a numerical claim about the unavailable MATLAB Statistics Toolbox.
    """
    data = np.maximum(np.asarray(data, dtype=np.float64), 0)
    rng = np.random.default_rng(seed)
    best_error = np.inf
    epsilon = np.finfo(np.float64).eps
    for _ in range(replicates):
        a, f = rng.random((len(data), materials)), rng.random((materials, data.shape[1]))
        for _ in range(multiplicative_iterations):
            f *= (a.T @ data) / np.maximum((a.T @ a) @ f, epsilon)
            a *= (data @ f.T) / np.maximum(a @ (f @ f.T), epsilon)
        error = np.linalg.norm(data - a @ f)
        if error < best_error:
            best_error, maps, spectra = error, a, f
    for _ in range(als_iterations):
        spectra = nonnegative_least_squares_small(maps.T @ maps, maps.T @ data)
        maps = nonnegative_least_squares_small(spectra @ spectra.T, (data @ spectra.T).T).T
    return maps, spectra


def tikhonov_cg(data, matrix, iterations=20, tolerance=1e-6, regularization=.001,
                backend='cpu'):
    """Independent ordinary CG per right-hand side for (W'W+lambda I)x=W'y."""
    op = LinearMap(cast_sparse_preserving_order(matrix, np.float64), backend)
    right = op.transpose(op.array(data))
    x = op.zeros(right.shape)
    residual = right.clone() if _torch(right) else right.copy()
    direction = residual.clone() if _torch(right) else residual.copy()
    residual_sq = (residual * residual).sum(axis=0)
    initial_sq = residual_sq.clone() if _torch(right) else residual_sq.copy()
    for _ in range(iterations):
        product = op.transpose(op.multiply(direction)) + regularization * direction
        denominator = (direction * product).sum(axis=0)
        if _torch(right):
            import torch
            active = residual_sq > tolerance ** 2 * initial_sq
            alpha = torch.where(active & (denominator != 0), residual_sq / denominator, 0)
        else:
            active = residual_sq > tolerance ** 2 * initial_sq
            alpha = np.divide(residual_sq, denominator, out=np.zeros_like(residual_sq),
                              where=active & (denominator != 0))
        x += direction * alpha[None, :]
        residual -= product * alpha[None, :]
        new_sq = (residual * residual).sum(axis=0)
        if _torch(right):
            beta = torch.where(active & (residual_sq != 0), new_sq / residual_sq, 0)
        else:
            beta = np.divide(new_sq, residual_sq, out=np.zeros_like(new_sq),
                             where=active & (residual_sq != 0))
        direction = residual + direction * beta[None, :]
        residual_sq = new_sq
    return op.numpy(x)


def sequential_comparison(data, matrix, order='RU', backend='cpu'):
    """Preserve reconstruction/unmixing comparison groups with explicit ports."""
    start = time.perf_counter()
    if order == 'RU':
        channels = np.maximum(tikhonov_cg(data, matrix, backend=backend), 0)
        maps, spectra = factor_nonnegative(channels)
    elif order == 'UR':
        projection_maps, spectra = factor_nonnegative(data)
        maps = tikhonov_cg(projection_maps, matrix, iterations=100, backend=backend)
    else:
        raise ValueError('order must be RU or UR')
    residual = np.linalg.norm(matrix @ maps @ spectra - data) / np.linalg.norm(data)
    return {'maps': maps, 'spectra': spectra, 'relative_residual': residual,
            'runtime_seconds': time.perf_counter() - start}
