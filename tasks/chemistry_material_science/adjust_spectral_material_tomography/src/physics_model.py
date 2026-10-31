"""Matched fan-beam line integration and explicit simultaneous iterative updates.

Line weights adapted from ASTRA 2.5.0 FanFlatBeamLineKernelProjector2D.inl.
Copyright 2010-2022 imec Vision Lab, University of Antwerp; 2014-2022 CWI.
GPL-3.0-or-later; full notice and extraction provenance: test/validation.md.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

IMAGE_SIZE = 255
DETECTOR_COUNT = 257
ANGLE_COUNT = 600
SOURCE_OBJECT_MM = 57.999001
SOURCE_DETECTOR_MM = 883.081116
DETECTOR_PITCH_MM = 0.80
CENTER_SHIFT = 257 / 2 - 127.379570


def geometry_vectors(angles: np.ndarray) -> np.ndarray:
    """Return (angle,6) source/central-detector/detector-step in image pixels."""
    theta = np.asarray(angles, dtype=np.float64)
    conversion = SOURCE_DETECTOR_MM / SOURCE_OBJECT_MM / DETECTOR_PITCH_MM
    sod = SOURCE_OBJECT_MM * conversion
    odd = (SOURCE_DETECTOR_MM - SOURCE_OBJECT_MM) * conversion
    pitch = DETECTOR_PITCH_MM * conversion
    st, ct = np.sin(theta), np.cos(theta)
    source = np.column_stack((sod * st, -sod * ct))
    step = np.column_stack((pitch * ct, pitch * st))
    detector = np.column_stack((-odd * st, odd * ct)) + CENTER_SHIFT * step
    return np.column_stack((source, detector, step))


def build_system_matrix(vectors: np.ndarray, size: int = IMAGE_SIZE,
                        detector_count: int = DETECTOR_COUNT) -> sparse.csr_matrix:
    """Explicit float32 line/pixel lengths, MATLAB angle-fast/image-column order.

    Coordinates are in image-pixel lengths. Each ray crosses constant-valued
    square pixels. Float32 accumulation preserves the selected upstream CPU
    discretization, including repeated-coordinate rounding along each ray.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    rows, cols, weights = [], [], []
    n_angle = len(vectors)
    vertical_rays = np.empty(n_angle * detector_count, dtype=bool)
    half = np.float32(size / 2 - 0.5)
    for a, vector in enumerate(vectors):
        sx, sy = np.float32(vector[:2])
        ux, uy = np.float32(vector[4:])
        dx0 = np.float32(vector[2] - detector_count * 0.5 * float(ux))
        dy0 = np.float32(vector[3] - detector_count * 0.5 * float(uy))
        det = np.arange(detector_count, dtype=np.float32) + np.float32(0.5)
        dx, dy = dx0 + det * ux, dy0 + det * uy
        rx, ry = sx - dx, sy - dy
        vertical = np.abs(rx) < np.abs(ry)
        vertical_rays[np.arange(detector_count) * n_angle + a] = vertical
        for is_vertical in (True, False):
            indices = np.flatnonzero(vertical == is_vertical)
            if not len(indices):
                continue
            if is_vertical:
                slope = rx[indices] / ry[indices]
                length = np.sqrt(rx[indices] ** 2 + ry[indices] ** 2) / np.abs(ry[indices])
                start = dx[indices] + (half - dy[indices]) * slope + half
            else:
                slope = ry[indices] / rx[indices]
                length = np.sqrt(rx[indices] ** 2 + ry[indices] ** 2) / np.abs(rx[indices])
                start = -(dy[indices] + (-half - dx[indices]) * slope - half)
            delta = -slope
            path = np.broadcast_to(delta[:, None], (len(indices), size)).copy()
            path[:, 0] = start
            coordinate = np.add.accumulate(path, axis=1, dtype=np.float32)
            nearest = np.floor(coordinate + np.float32(0.5)).astype(np.int32)
            offset = coordinate - nearest.astype(np.float32)
            s = np.float32(0.5) - np.float32(0.5) * np.abs(slope)
            t = np.float32(0.5) + np.float32(0.5) * np.abs(slope)
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = length / (t - s)
                left = offset < -s[:, None]
                right = offset > s[:, None]
                central = np.broadcast_to(length[:, None], coordinate.shape).copy()
                central[left] = ((offset + t[:, None]) * factor[:, None])[left]
                central[right] = (length[:, None] - (offset - s[:, None]) * factor[:, None])[right]
            neighbor = np.where(left, nearest - 1, nearest + 1)
            neighbor_weight = length[:, None] - central
            neighbor_weight[right] = ((offset - s[:, None]) * factor[:, None])[right]
            along = np.broadcast_to(np.arange(size), coordinate.shape)
            ray_rows = np.broadcast_to((indices * n_angle + a)[:, None], coordinate.shape)
            for transverse, value, enabled in (
                (nearest, central, np.ones(coordinate.shape, dtype=bool)),
                (neighbor, neighbor_weight, left | right),
            ):
                keep = enabled & (transverse >= 0) & (transverse < size)
                r, c = (along, transverse) if is_vertical else (transverse, along)
                rows.append(ray_rows[keep].astype(np.int32))
                cols.append((r[keep] + size * c[keep]).astype(np.int32))
                weights.append(value[keep].astype(np.float32))
    matrix = sparse.coo_matrix((np.concatenate(weights),
                                (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n_angle * detector_count, size * size)).tocsr()
    return preserve_source_pixel_order(matrix, size, vertical_rays)


def preserve_source_pixel_order(matrix: sparse.csr_matrix, size: int,
                                vertical_rays: np.ndarray) -> sparse.csr_matrix:
    """Retain source dominant-axis traversal with MATLAB column labels.

    Vertical rays visit rows then columns; horizontal rays visit columns then
    rows. The column labels are row + size*column. CSR sorting changes roundoff in
    long iterative trajectories even though the mathematical matrix is equal.
    This routine permutes weights within each row, without changing any value.
    """
    for ray in np.flatnonzero(vertical_rays):
        start, stop = matrix.indptr[ray:ray + 2]
        columns = matrix.indices[start:stop]
        source_columns = (columns % size) * size + columns // size
        order = np.argsort(source_columns, kind='stable')
        matrix.indices[start:stop] = columns[order]
        matrix.data[start:stop] = matrix.data[start:stop][order]
    matrix.has_sorted_indices = False
    return matrix


def forward(matrix: sparse.spmatrix, maps: np.ndarray) -> np.ndarray:
    """Project (pixel,quantity) images into (ray,quantity) line integrals."""
    return matrix @ maps


def adjoint(matrix: sparse.spmatrix, sinograms: np.ndarray) -> np.ndarray:
    """Apply the exact transpose of the declared acquisition discretization."""
    return matrix.T @ sinograms


def sirt(matrix: sparse.spmatrix, data: np.ndarray, iterations: int = 100,
         backend: str = 'cpu', verbose: bool = False) -> np.ndarray:
    """Unconstrained row/column-normalized SIRT, zero initialization, float32."""
    from .solvers import LinearMap, cast_sparse_preserving_order
    if backend in ('cpu', 'cuda_ordered'):
        return source_ordered_sirt(matrix, data, iterations, verbose=verbose,
                                   backend=backend)
    op = LinearMap(cast_sparse_preserving_order(matrix, np.float32), backend=backend)
    y = op.array(np.asarray(data, dtype=np.float32))
    row = op.array(np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32))
    col = op.array(np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32))
    if backend == 'cuda':
        import torch
        row = torch.where(row > 0, 1 / row, 0)
        col = torch.where(col > 0, 1 / col, 0)
    else:
        row = np.divide(1, row, out=np.zeros_like(row), where=row > 0)
        col = np.divide(1, col, out=np.zeros_like(col), where=col > 0)
    x = op.zeros((matrix.shape[1], data.shape[1]))
    for _ in range(iterations):
        x += col[:, None] * op.transpose(row[:, None] * (y - op.multiply(x)))
    return op.numpy(x)


def source_ordered_sirt(matrix, data, iterations=100, verbose=False, backend='cpu'):
    """Evaluate the CPU source's ordered float32 residual and backprojection.

    Bound full geometry arrives in MATLAB ray/pixel order. Convert labels to
    original detector-fast rays and row-major pixels without sorting ray entries.
    Smaller mathematical cases retain their caller-supplied coordinate order.
    """
    from .solvers import cast_sparse_preserving_order
    if backend not in ('cpu', 'cuda_ordered'):
        raise ValueError('source-ordered SIRT backend must be cpu or cuda_ordered')
    w = cast_sparse_preserving_order(matrix, np.float32)
    y = np.asarray(data, dtype=np.float32)
    bound_geometry = w.shape == (600 * 257, 255 * 255)
    if bound_geometry:
        rays = np.arange(600 * 257).reshape(257, 600).T.ravel()
        pixels = np.arange(255 * 255).reshape(255, 255).T.ravel()
        w = w[rays].tocsr()
        w.indices = pixels[w.indices]
        w.has_sorted_indices = False
        y = y[rays]
    transposed = w.T.tocsr()
    m, n = w.shape
    row = np.asarray(w @ np.ones(n, dtype=np.float32))
    col = np.asarray(transposed @ np.ones(m, dtype=np.float32))
    row = np.divide(1, row, out=np.zeros_like(row), where=abs(row) > 1e-6)
    col = np.divide(1, col, out=np.zeros_like(col), where=abs(col) > 1e-6)
    # [I,-W] starts each residual at y and subtracts each source-ordered term.
    indptr = w.indptr + np.arange(m + 1)
    starts = indptr[:-1]
    positions = np.arange(w.nnz) + np.repeat(np.arange(m) + 1, np.diff(w.indptr))
    indices = np.empty(w.nnz + m, dtype=np.int32)
    values = np.empty(w.nnz + m, dtype=np.float32)
    indices[starts], values[starts] = np.arange(m), 1
    indices[positions], values[positions] = w.indices + m, -w.data
    augmented = sparse.csr_matrix((values, indices, indptr), shape=(m, m + n))
    del positions, indices, values, w
    combined = np.vstack((y, np.zeros((n, y.shape[1]), dtype=np.float32)))
    x = combined[m:]
    operators = []
    try:
        if backend == 'cuda_ordered':
            from .deterministic_sparse import OrderedCsr
            operators = [OrderedCsr(augmented), OrderedCsr(transposed)]
            # Host buffers must outlive any queued host-to-device copies.
            operators[0].cp.cuda.get_current_stream().synchronize()
            del augmented, transposed
        for iteration in range(iterations):
            residual = operators[0].multiply(combined) if operators else augmented @ combined
            weighted = row[:, None] * residual
            correction = operators[1].multiply(weighted) if operators else transposed @ weighted
            x += col[:, None] * correction
            if verbose and (iteration == 0 or (iteration + 1) % 10 == 0):
                print(f'{backend} SIRT {iteration+1}/{iterations}: all {y.shape[1]} energy channels', flush=True)
    finally:
        for operator in operators:
            operator.close()
    return x[pixels] if bound_geometry else x
