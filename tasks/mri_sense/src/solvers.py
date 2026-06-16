"""
CG-SENSE Reconstruction Solver
================================

Implements Conjugate Gradient SENSE for multi-coil MRI reconstruction.
CG-SENSE solves the normal equations of the SENSE encoding model:

    A^H A x = A^H b

where A is the SENSE encoding operator (image -> undersampled multi-coil
k-space) and b is the acquired undersampled k-space data.

The normal equations are solved using the Conjugate Gradient algorithm,
which iteratively minimizes ||A x - b||^2 via search along conjugate
directions.

Key difference from GRAPPA: CG-SENSE works in the image domain by
directly solving the linear system, while GRAPPA works in k-space by
interpolating missing samples. CG-SENSE can handle arbitrary sampling
patterns and naturally incorporates coil sensitivity information.

Reference
---------
Pruessmann et al., "Advances in sensitivity encoding with arbitrary
k-space trajectories," MRM 46.4 (2001): 638-651.

CG algorithm ported from scipy.sparse.linalg.cg (scipy/sparse/linalg/
_isolve/iterative.py).

Implementation adapted from pygrappa (mckib2/pygrappa).
"""

import numpy as np

from src.physics_model import centered_fft2, centered_ifft2


def conjugate_gradient(matvec, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None):
    """Conjugate Gradient solver for Hermitian positive-definite systems.

    Solves A x = b where A is given as a function (matvec).
    Handles complex-valued systems using conjugate inner products.

    Ported from scipy.sparse.linalg.cg.

    Parameters
    ----------
    matvec : callable
        Function computing A @ x. Must accept and return 1D arrays.
    b : ndarray, (n,) complex or real
        Right-hand side vector.
    x0 : ndarray or None
        Initial guess. If None, starts from zeros.
    rtol : float
        Relative tolerance: converges when ||r|| < rtol * ||b||.
    atol : float
        Absolute tolerance: converges when ||r|| < atol.
    maxiter : int or None
        Maximum iterations. Default: 10 * len(b).

    Returns
    -------
    x : ndarray, (n,)
        Solution vector.
    info : int
        0 if converged, maxiter if not.
    """
    n = len(b)
    if maxiter is None:
        maxiter = n * 10

    bnrm2 = np.linalg.norm(b)
    tol = max(atol, rtol * bnrm2)

    if bnrm2 == 0:
        return b.copy(), 0

    # Use conjugate inner product for complex, plain dot for real
    dotprod = np.vdot if np.iscomplexobj(b) else np.dot

    # Initialize
    if x0 is not None:
        x = x0.copy()
        r = b - matvec(x)
    else:
        x = np.zeros_like(b)
        r = b.copy()

    rho_prev = None
    p = None

    for iteration in range(maxiter):
        # Convergence check
        if np.linalg.norm(r) < tol:
            return x, 0

        # No preconditioner: z = r (standard CG)
        z = r

        rho_cur = dotprod(r, z)

        if iteration > 0:
            beta = rho_cur / rho_prev
            p = z + beta * p
        else:
            p = z.copy()

        # Matrix-vector product
        q = matvec(p)

        # Step size
        alpha = rho_cur / dotprod(p, q)

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * q

        rho_prev = rho_cur

    # Did not converge
    return x, maxiter


def cgsense_reconstruct(
    kspace_us: np.ndarray,
    sens: np.ndarray,
    coil_axis: int = -1,
) -> np.ndarray:
    """
    Reconstruct a single-coil image from undersampled multi-coil
    k-space using Conjugate Gradient SENSE.

    Parameters
    ----------
    kspace_us : ndarray, (Nx, Ny, Nc) complex128
        Undersampled multi-coil k-space (zeros at missing locations).
    sens : ndarray, (Nx, Ny, Nc) complex128
        Coil sensitivity maps.
    coil_axis : int
        Dimension holding coil data.

    Returns
    -------
    recon : ndarray, (Nx, Ny) complex128
        Reconstructed single-coil image.
    """
    # Ensure coils are last
    kspace = np.moveaxis(kspace_us, coil_axis, -1)
    sens = np.moveaxis(sens, coil_axis, -1)

    dims = kspace.shape[:-1]
    mask = np.abs(kspace[..., 0]) > 0

    # A: image (Nx*Ny,) -> undersampled multi-coil k-space (Nx*Ny*Nc,)
    def _A(x0):
        x = np.reshape(x0, dims)
        res = centered_fft2(x[..., None] * sens) * mask[..., None]
        return res.reshape(-1)

    # A^H: k-space (Nx*Ny*Nc,) -> image (Nx*Ny,)
    def _AH(y0):
        y = np.reshape(y0, kspace.shape)
        res = np.sum(sens.conj() * centered_ifft2(y), axis=-1)
        return res.reshape(-1)

    # E = A^H A: (Nx*Ny,) -> (Nx*Ny,)
    def E(x0):
        return _AH(_A(x0))

    b = _AH(kspace.reshape(-1))

    x, info = conjugate_gradient(E, b, atol=0)

    return np.reshape(x, dims)


def cgsense_image_recon(
    kspace_us: np.ndarray,
    sens: np.ndarray,
) -> np.ndarray:
    """
    Full CG-SENSE pipeline: solve encoding equation, return magnitude.

    Parameters
    ----------
    kspace_us : (Nx, Ny, Nc) complex128
    sens : (Nx, Ny, Nc) complex128

    Returns
    -------
    recon : (Nx, Ny) float64
        Normalized magnitude image in [0, 1].
    """
    x = cgsense_reconstruct(kspace_us, sens, coil_axis=-1)
    mag = np.abs(x)
    return mag / mag.max() if mag.max() > 0 else mag
