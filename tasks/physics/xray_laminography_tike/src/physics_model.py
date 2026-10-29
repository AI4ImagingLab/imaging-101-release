"""Physics model for X-ray laminography/tomography.

Implements the laminographic forward projection and adjoint using the
Fourier slice theorem with an unequally-spaced FFT (USFFT) approach.

The forward model computes 2D projections of a 3D volume at each rotation
angle theta with a given tilt angle. For tilt = pi/2, this reduces to
standard parallel-beam tomography.

No dependency on tike. Uses cupy for GPU acceleration and its built-in
FFT and scatter/gather operations.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def _get_xp(arr):
    """Return cupy if arr is a cupy array, else numpy."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def make_frequency_grid(theta, tilt, n, xp=None):
    """Compute the 3D frequency coordinates for laminographic projection.

    For each rotation angle and each 2D frequency (ku, kv), compute the
    corresponding 3D frequency point in the rotated/tilted frame.

    The coordinate transform follows tike's convention (grid.cu):
        f[0] = kv * sin(tilt)
        f[1] = -ku * sin(theta) + kv * cos(theta) * cos(tilt)
        f[2] = +ku * cos(theta) + kv * sin(theta) * cos(tilt)

    Parameters
    ----------
    theta : (R,) float32
        Rotation angles in radians.
    tilt : float
        Tilt angle in radians. pi/2 for standard tomography.
    n : int
        Grid size.
    xp : module
        numpy or cupy.

    Returns
    -------
    xi : (R*n*n, 3) float32
        Unequally-spaced frequency coordinates.
    """
    if xp is None:
        xp = np

    R = theta.shape[0]
    ctilt = float(np.cos(tilt))
    stilt = float(np.sin(tilt))

    # Frequency indices in [-0.5, 0.5)
    k = (xp.arange(n, dtype=xp.float32) - n // 2) / n  # (n,)

    # Build grid for all angles
    xi = xp.empty((R * n * n, 3), dtype=xp.float32)

    for p in range(R):
        ctheta = float(xp.cos(theta[p]))
        stheta = float(xp.sin(theta[p]))
        offset = p * n * n

        for y in range(n):
            kv = float(k[y])
            for x in range(n):
                ku = float(k[x])
                idx = offset + y * n + x
                xi[idx, 0] = kv * stilt
                xi[idx, 1] = -ku * stheta + kv * ctheta * ctilt
                xi[idx, 2] = ku * ctheta + kv * stheta * ctilt

    return xi


def _make_frequency_grid_fast(theta, tilt, n, xp):
    """Vectorized version of make_frequency_grid for GPU performance."""
    R = theta.shape[0]
    ctilt = xp.float32(np.cos(float(tilt)))
    stilt = xp.float32(np.sin(float(tilt)))

    k = (xp.arange(n, dtype=xp.float32) - n // 2) / n

    # (R,) trig values
    ctheta = xp.cos(theta)  # (R,)
    stheta = xp.sin(theta)  # (R,)

    # Build meshgrid for ku, kv
    ku, kv = xp.meshgrid(k, k, indexing='xy')  # both (n, n)
    ku = ku.ravel()  # (n*n,)
    kv = kv.ravel()  # (n*n,)

    # Broadcast: (R, n*n, 3)
    xi = xp.empty((R, n * n, 3), dtype=xp.float32)
    xi[:, :, 0] = kv[None, :] * stilt
    xi[:, :, 1] = (-ku[None, :] * stheta[:, None]
                    + kv[None, :] * ctheta[:, None] * ctilt)
    xi[:, :, 2] = (ku[None, :] * ctheta[:, None]
                    + kv[None, :] * stheta[:, None] * ctilt)

    return xi.reshape(R * n * n, 3)


def _checkerboard(x, axes, xp):
    """Multiply array by (-1)^(i+j) along specified axes for FFT shifting."""
    for ax in axes:
        s = x.shape[ax]
        phase = xp.ones(s, dtype=x.real.dtype)
        phase[1::2] = -1
        view_shape = [1] * x.ndim
        view_shape[ax] = s
        x = x * phase.reshape(view_shape)
    return x


def _nufft_forward(u, xi, n, xp):
    """Equally-spaced to unequally-spaced 3D FFT via interpolation.

    This is a simplified NUFFT (type 2): given a 3D volume u on a regular grid,
    evaluate its Fourier transform at arbitrary frequency points xi.

    Uses nearest-neighbor interpolation for simplicity. For production quality,
    a proper NUFFT library (e.g., cufinufft) would be used.

    Parameters
    ----------
    u : (n, n, n) complex64
        Input volume.
    xi : (M, 3) float32
        Target frequencies in [-0.5, 0.5).

    Returns
    -------
    F : (M,) complex64
        Fourier values at target frequencies.
    """
    # Compute full 3D FFT of the volume
    U = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(u)))  # centered FFT

    # Map frequencies to grid indices
    # xi is in [-0.5, 0.5), grid indices are [0, n)
    grid_coords = (xi + 0.5) * n  # (M, 3), in [0, n)

    # Nearest-neighbor interpolation
    idx = xp.clip(xp.round(grid_coords).astype(xp.int32), 0, n - 1)
    F = U[idx[:, 0], idx[:, 1], idx[:, 2]]

    return F


def _nufft_adjoint(F, xi, n, xp):
    """Unequally-spaced to equally-spaced 3D FFT (adjoint/type 1 NUFFT).

    Given values at arbitrary frequency points, accumulate them onto
    a regular 3D grid and inverse FFT.

    Parameters
    ----------
    F : (M,) complex64
        Values at unequally-spaced frequencies.
    xi : (M, 3) float32
        Frequency coordinates in [-0.5, 0.5).
    n : int
        Grid size.

    Returns
    -------
    u : (n, n, n) complex64
        Volume recovered from frequency data.
    """
    # Map frequencies to grid indices
    grid_coords = (xi + 0.5) * n
    idx = xp.clip(xp.round(grid_coords).astype(xp.int32), 0, n - 1)

    # Accumulate onto grid (scatter-add)
    # CuPy scatter_add doesn't support complex types, so split real/imag
    U_real = xp.zeros((n, n, n), dtype=xp.float32)
    U_imag = xp.zeros((n, n, n), dtype=xp.float32)
    indices = (idx[:, 0], idx[:, 1], idx[:, 2])
    if xp is np:
        np.add.at(U_real, indices, F.real.astype(np.float32))
        np.add.at(U_imag, indices, F.imag.astype(np.float32))
    else:
        import cupyx
        cupyx.scatter_add(U_real, indices, F.real.astype(xp.float32))
        cupyx.scatter_add(U_imag, indices, F.imag.astype(xp.float32))
    U = (U_real + 1j * U_imag).astype(xp.complex64)

    # Inverse centered FFT
    u = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(U)))

    return u


def forward_project(obj, theta, tilt):
    """Compute laminographic projections of a 3D volume.

    Implements the forward laminography operator using the Fourier slice
    theorem: the 2D Fourier transform of a projection equals a slice through
    the 3D Fourier transform of the volume, with the slice orientation
    determined by the rotation angle and tilt.

    Parameters
    ----------
    obj : (n, n, n) complex64
        The 3D volume.
    theta : (R,) float32
        Rotation angles in radians.
    tilt : float
        Tilt angle in radians. pi/2 for standard tomography.

    Returns
    -------
    data : (R, n, n) complex64
        Simulated projection data.
    """
    xp = _get_xp(obj)
    obj = xp.asarray(obj, dtype=xp.complex64)
    theta = xp.asarray(theta, dtype=xp.float32)

    assert obj.ndim == 3, f"obj must be 3D, got shape {obj.shape}"
    assert theta.ndim == 1, f"theta must be 1D, got shape {theta.shape}"

    n = obj.shape[0]
    R = theta.shape[0]

    # Compute frequency grid
    xi = _make_frequency_grid_fast(theta, tilt, n, xp)

    # Forward NUFFT: volume → Fourier values at tilted/rotated frequencies
    F = _nufft_forward(obj, xi, n, xp)
    F = F.reshape(R, n, n)

    # Apply checkerboard phase correction and inverse 2D FFT
    F = _checkerboard(F, axes=(1, 2), xp=xp)
    data = xp.fft.ifft2(F, axes=(1, 2), norm='ortho')
    data = _checkerboard(data, axes=(1, 2), xp=xp)

    return data.astype(xp.complex64)


def adjoint_project(data, theta, tilt, n):
    """Compute the adjoint laminographic operator (backprojection).

    Parameters
    ----------
    data : (R, n, n) complex64
        Projection data.
    theta : (R,) float32
        Rotation angles.
    tilt : float
        Tilt angle.
    n : int
        Volume grid size.

    Returns
    -------
    obj : (n, n, n) complex64
        Backprojected volume.
    """
    xp = _get_xp(data)
    data = xp.asarray(data, dtype=xp.complex64)
    theta = xp.asarray(theta, dtype=xp.float32)
    R = theta.shape[0]

    # Forward 2D FFT with checkerboard
    F = _checkerboard(data.copy(), axes=(1, 2), xp=xp)
    F = xp.fft.fft2(F, axes=(1, 2), norm='ortho')
    F = _checkerboard(F, axes=(1, 2), xp=xp)

    # Compute frequency grid
    xi = _make_frequency_grid_fast(theta, tilt, n, xp)

    # Adjoint NUFFT: scatter Fourier values back to 3D grid
    # Note: adjoint uses -xi (conjugate transpose)
    obj = _nufft_adjoint(F.ravel(), -xi, n, xp)
    obj /= n ** 2

    return obj.astype(xp.complex64)


def cost_function(obj, data, theta, tilt):
    """Least-squares cost: ||forward(obj) - data||^2."""
    xp = _get_xp(obj)
    residual = forward_project(obj, theta, tilt) - data
    return float(xp.sum(xp.abs(residual).astype(xp.float32) ** 2).real)


def gradient(obj, data, theta, tilt):
    """Gradient of the least-squares cost function.

    grad = adjoint(forward(obj) - data) / (R * n^3)
    """
    xp = _get_xp(obj)
    n = obj.shape[0]
    R = data.shape[0]
    residual = forward_project(obj, theta, tilt) - data
    grad = adjoint_project(residual, theta, tilt, n)
    grad /= (R * n ** 3)
    return grad
