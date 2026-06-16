"""
Physics model for CT imaging: Radon transform and filtered back projection.

The CT forward model is the Radon transform — line integrals of the image
along rays at specified angles:

    g(theta, s) = integral f(s*cos(theta) - t*sin(theta),
                              s*sin(theta) + t*cos(theta)) dt

The adjoint/approximate inverse is filtered back projection (FBP):
    1. Apply ramp filter to each projection in the Fourier domain
    2. Back-project: for each angle, smear the filtered projection back
       along the ray direction and accumulate

Implementation ported from scikit-image radon_transform.py (Kak & Slaney 1988):
- radon(): rotate image by each angle via bilinear interpolation, sum columns
- iradon(): ramp filter in Fourier domain, back-project via interpolation

Uses only numpy (fft, interp, coordinate transforms). No scipy dependency.

Reference
---------
Kak, A.C. & Slaney, M. (1988). Principles of Computerized Tomographic
Imaging. IEEE Press.

scikit-image: skimage.transform.radon_transform (Romberg implementation)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Image rotation via affine coordinate mapping + interpolation
# Ported from scikit-image radon_transform.py (lines 103-113) and
# _warps.py warp() which uses scipy.ndimage.map_coordinates for
# bilinear interpolation at the mapped coordinates.
# ---------------------------------------------------------------------------

def _rotate_image(image, angle_rad):
    """
    Rotate a 2D image by the given angle using bilinear interpolation.

    Rotation is about the image center. Pixels outside the original image
    are set to zero (zero-padding, mode='constant', cval=0).

    Ported from scikit-image's radon():
    - The 3x3 affine rotation matrix R (radon_transform.py lines 105-108)
      maps output pixel (row, col) to input pixel coordinates.
    - scipy.ndimage.map_coordinates performs order-1 (bilinear) interpolation
      at the mapped coordinates, matching scikit-image's warp(clip=False).

    Parameters
    ----------
    image : ndarray, (N, N)
        Square input image.
    angle_rad : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    rotated : ndarray, (N, N)
        Rotated image.
    """
    from scipy.ndimage import map_coordinates

    N = image.shape[0]
    center = N // 2

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Affine rotation matrix R from scikit-image radon_transform.py lines 105-108:
    #   R = [[cos_a,  sin_a, -center*(cos_a + sin_a - 1)],
    #        [-sin_a, cos_a, -center*(cos_a - sin_a - 1)],
    #        [0,      0,      1                          ]]
    # This maps output (col, row) -> input (col, row).
    # For map_coordinates we need input (row, col) for each output pixel.

    rows, cols = np.mgrid[0:N, 0:N].astype(np.float64)

    src_col = cos_a * cols + sin_a * rows + (-center * (cos_a + sin_a - 1))
    src_row = -sin_a * cols + cos_a * rows + (-center * (cos_a - sin_a - 1))

    # map_coordinates expects coordinates as (row_coords, col_coords)
    coords = np.array([src_row, src_col])

    rotated = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)

    return rotated


# ---------------------------------------------------------------------------
# Radon transform (forward projection)
# ---------------------------------------------------------------------------

def radon_transform(image, angles_deg):
    """
    Compute the Radon transform (sinogram) of a 2D image.

    For each angle, rotates the image and sums along columns (axis=0),
    giving the line integral along parallel rays at that angle.

    Assumes the image is zero outside the inscribed circle (circle=True
    convention from scikit-image).

    Parameters
    ----------
    image : ndarray, (N, N)
        Input square image.
    angles_deg : ndarray, (n_angles,)
        Projection angles in degrees.

    Returns
    -------
    sinogram : ndarray, (N, n_angles)
        Radon transform. sinogram[:, i] is the projection at angles_deg[i].
    """
    image = np.asarray(image, dtype=np.float64)
    N = image.shape[0]

    # Enforce circle constraint: zero outside inscribed circle
    center = N // 2
    radius = N // 2
    y, x = np.ogrid[:N, :N]
    outside = ((y - center) ** 2 + (x - center) ** 2) > radius ** 2
    image = image.copy()
    image[outside] = 0.0

    sinogram = np.zeros((N, len(angles_deg)), dtype=np.float64)

    for i, angle in enumerate(np.deg2rad(angles_deg)):
        rotated = _rotate_image(image, angle)
        sinogram[:, i] = rotated.sum(axis=0)

    return sinogram


# ---------------------------------------------------------------------------
# Ramp filter for FBP (ported from scikit-image _get_fourier_filter)
# ---------------------------------------------------------------------------

def _get_ramp_filter(size, filter_name="ramp"):
    """
    Construct the Fourier-domain ramp filter for FBP.

    Uses the Ram-Lak construction (spatial-domain impulse response → FFT)
    to reduce discretization artifacts, as described in Kak & Slaney Ch. 3.

    Parameters
    ----------
    size : int
        Filter size (must be even, typically next power of 2).
    filter_name : str
        'ramp' or None. If None, returns all-ones (no filtering).

    Returns
    -------
    fourier_filter : ndarray, (size, 1)
    """
    if filter_name is None:
        return np.ones((size, 1))

    # Spatial-domain ramp filter impulse response (Kak & Slaney Eq. 61)
    n = np.concatenate((
        np.arange(1, size / 2 + 1, 2, dtype=int),
        np.arange(size / 2 - 1, 0, -2, dtype=int),
    ))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    fourier_filter = 2 * np.real(np.fft.fft(f))
    return fourier_filter[:, np.newaxis]


# ---------------------------------------------------------------------------
# Filtered Back Projection (iradon)
# ---------------------------------------------------------------------------

def filtered_back_projection(sinogram, angles_deg, output_size=None,
                              filter_name="ramp"):
    """
    Reconstruct an image from a sinogram using filtered back projection.

    Algorithm:
    1. Pad sinogram to next power of 2
    2. Apply ramp filter in Fourier domain (frequency-domain filtering)
    3. For each angle, back-project the filtered projection: interpolate
       the 1D projection onto the 2D image grid using the geometry
       t = y*cos(angle) - x*sin(angle)

    Ported from scikit-image iradon() with circle=True convention.

    Parameters
    ----------
    sinogram : ndarray, (n_det, n_angles)
        Sinogram (Radon transform data).
    angles_deg : ndarray, (n_angles,)
        Projection angles in degrees.
    output_size : int or None
        Size of output image. If None, uses sinogram height.
    filter_name : str or None
        'ramp' for standard FBP, None for unfiltered back-projection.

    Returns
    -------
    reconstruction : ndarray, (output_size, output_size)
    """
    sinogram = np.asarray(sinogram, dtype=np.float64)
    n_det = sinogram.shape[0]
    n_angles = len(angles_deg)

    if output_size is None:
        output_size = n_det

    # Circle mode: pad sinogram to diagonal length
    diagonal = int(np.ceil(np.sqrt(2) * n_det))
    pad = diagonal - n_det
    old_center = n_det // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    sinogram_padded = np.pad(
        sinogram, ((pad_before, pad - pad_before), (0, 0)),
        mode='constant', constant_values=0,
    )
    img_shape = sinogram_padded.shape[0]

    # Pad to next power of 2 for FFT efficiency
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(sinogram_padded, pad_width, mode='constant', constant_values=0)

    # Apply ramp filter in Fourier domain
    fourier_filter = _get_ramp_filter(projection_size_padded, filter_name)
    projection = np.fft.fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(np.fft.ifft(projection, axis=0))[:img_shape, :]

    # Back-project: for each angle, interpolate filtered projection onto image grid
    reconstructed = np.zeros((output_size, output_size), dtype=np.float64)
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    det_coords = np.arange(img_shape) - img_shape // 2

    for col_idx, angle in enumerate(np.deg2rad(angles_deg)):
        # Project pixel coordinates onto detector: t = y*cos - x*sin
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        # Interpolate the filtered projection at these t values
        reconstructed += np.interp(t.ravel(), det_coords, radon_filtered[:, col_idx],
                                    left=0, right=0).reshape(output_size, output_size)

    # Apply circle mask
    out_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
    reconstructed[out_circle] = 0.0

    return reconstructed * np.pi / (2 * n_angles)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def add_gaussian_noise(sinogram, noise_std, rng=None):
    """Add Gaussian noise to a sinogram.

    Parameters
    ----------
    sinogram : ndarray
        Clean sinogram.
    noise_std : float
        Standard deviation relative to sinogram max.
    rng : np.random.Generator or None

    Returns
    -------
    noisy_sinogram : ndarray
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noise_level = noise_std * np.max(np.abs(sinogram))
    noise = rng.normal(0, noise_level, size=sinogram.shape)
    return sinogram + noise
