"""Forward model for parallel-beam X-ray CT.

Implements the Radon transform (forward projection) and its adjoint
(back-projection) for parallel-beam geometry. The forward model maps
a 2-D attenuation image to sinogram data via line integrals.
"""

import numpy as np
from scipy.ndimage import map_coordinates


class ParallelBeamProjector:
    """Parallel-beam X-ray CT forward and adjoint operators.

    The forward operator computes the Radon transform of a 2-D image:
    for each projection angle theta, the detector measures line integrals
    of the attenuation coefficient along parallel rays.

    Parameters
    ----------
    n_pixels : int
        Size of the square reconstruction grid (n_pixels x n_pixels).
    n_detector : int
        Number of detector pixels per projection.
    theta : ndarray, shape (n_angles,)
        Projection angles in radians.
    """

    def __init__(self, n_pixels, n_detector, theta):
        self.n_pixels = n_pixels
        self.n_detector = n_detector
        self.theta = np.asarray(theta, dtype=np.float64)
        self.n_angles = len(self.theta)

    def forward(self, image):
        """Compute the Radon transform (forward projection).

        Parameters
        ----------
        image : ndarray, shape (n_pixels, n_pixels)
            2-D attenuation coefficient map.

        Returns
        -------
        sinogram : ndarray, shape (n_angles, n_detector)
            Sinogram (projection data).
        """
        image = np.asarray(image, dtype=np.float64)
        n = self.n_pixels
        center = (n - 1) / 2.0
        det_center = (self.n_detector - 1) / 2.0

        sinogram = np.zeros((self.n_angles, self.n_detector), dtype=np.float64)

        for i, angle in enumerate(self.theta):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            for d in range(self.n_detector):
                t = d - det_center
                # Ray parameterised by s: (x, y) = t*(cos, sin) + s*(-sin, cos)
                s_vals = np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
                x_coords = t * cos_a - s_vals * sin_a + center
                y_coords = t * sin_a + s_vals * cos_a + center

                coords = np.array([y_coords, x_coords])
                line_integral = map_coordinates(
                    image, coords, order=1, mode='constant', cval=0.0
                )
                sinogram[i, d] = np.sum(line_integral)

        return sinogram

    def adjoint(self, sinogram):
        """Compute the back-projection (adjoint of the Radon transform).

        Parameters
        ----------
        sinogram : ndarray, shape (n_angles, n_detector)
            Sinogram data.

        Returns
        -------
        image : ndarray, shape (n_pixels, n_pixels)
            Back-projected image.
        """
        sinogram = np.asarray(sinogram, dtype=np.float64)
        n = self.n_pixels
        center = (n - 1) / 2.0
        det_center = (self.n_detector - 1) / 2.0

        image = np.zeros((n, n), dtype=np.float64)

        y_grid, x_grid = np.mgrid[0:n, 0:n]
        x_grid = x_grid.astype(np.float64) - center
        y_grid = center - y_grid.astype(np.float64)

        for i, angle in enumerate(self.theta):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            t = x_grid * cos_a + y_grid * sin_a + det_center
            image += np.interp(t.ravel(), np.arange(self.n_detector),
                               sinogram[i]).reshape(n, n)

        return image


def find_rotation_center(sinogram, theta, init=None, tol=0.5):
    """Find the rotation center by minimizing reconstruction variance.

    Uses a coarse grid search followed by refinement to find the center
    of rotation that produces the sharpest (highest-variance)
    reconstruction, equivalent to minimizing reconstruction artifacts.

    Parameters
    ----------
    sinogram : ndarray, shape (n_angles, n_detector)
        Preprocessed sinogram (single slice).
    theta : ndarray, shape (n_angles,)
        Projection angles in radians.
    init : float, optional
        Initial guess for rotation center (in detector pixels).
        Defaults to the center of the detector.
    tol : float
        Convergence tolerance in pixels.

    Returns
    -------
    float
        Estimated rotation center in detector pixel units.
    """
    n_det = sinogram.shape[1]
    if init is None:
        init = n_det / 2.0

    # Use cross-correlation of opposite projections to find center.
    # For parallel-beam CT with theta spanning [0, pi), the projection
    # at angle theta is the mirror of the projection at angle theta + pi.
    # The first and last projections approximate this pair.
    p_first = sinogram[0, :]
    p_last = sinogram[-1, ::-1]

    # Cross-correlate to find the shift
    from scipy.signal import correlate
    cc = correlate(p_first, p_last, mode='full')
    shift_range = np.arange(-(n_det - 1), n_det)
    peak_shift = shift_range[np.argmax(cc)]

    # The rotation center is at (n_det - 1) / 2 + shift / 2
    center_estimate = (n_det - 1) / 2.0 + peak_shift / 2.0

    # Refine with sub-pixel search using reconstruction quality
    def metric(center):
        shifted = _shift_sinogram(sinogram, center, n_det)
        recon = _fbp_slice(shifted, theta, n_det)
        return -np.var(recon)

    fine_step = tol
    centers_fine = np.arange(
        center_estimate - 5.0, center_estimate + 5.0, fine_step
    )
    values_fine = np.array([metric(c) for c in centers_fine])
    return centers_fine[np.argmin(values_fine)]


def _shift_sinogram(sinogram, center, n_det):
    """Shift sinogram so that the rotation center is at the detector midpoint."""
    shift = center - (n_det - 1) / 2.0
    if abs(shift) < 1e-6:
        return sinogram
    from scipy.ndimage import shift as ndshift
    return ndshift(sinogram, [0, -shift], mode='constant', cval=0.0)


def _fbp_slice(sinogram, theta, n_det):
    """Quick FBP for a single sinogram slice (used in center-finding)."""
    from src.solvers import filtered_back_projection
    return filtered_back_projection(sinogram, theta, n_det)
