"""
Forward model and measurement functions for dynamic black hole imaging.

Extracted from ehtim's StarWarps implementation (starwarps.py).
All ehtim dependencies removed - uses plain numpy arrays.
"""

import numpy as np
import scipy.linalg


def delta_pulse_2d(ux, uy, psize, dom="F"):
    """Delta pulse - always returns 1.0."""
    return 1.0


class DFTForwardModel:
    """Discrete Fourier Transform forward model for interferometry.

    Builds a DFT matrix A that maps image pixels to visibility measurements.
    """

    def __init__(self, uv_coords, N, pixel_size_rad):
        """
        Args:
            uv_coords: (M, 2) array of (u, v) baseline coordinates
            N: image dimension (N x N pixels)
            pixel_size_rad: pixel size in radians
        """
        self.uv_coords = uv_coords
        self.N = N
        self.pixel_size_rad = pixel_size_rad
        self._A = self._build_dft_matrix()

    def _build_dft_matrix(self):
        """Build DFT matrix A of shape (M, N²)."""
        M = len(self.uv_coords)
        npixels = self.N * self.N

        # Generate pixel coordinates
        xlist = np.arange(0, -self.N, -1) * self.pixel_size_rad + \
                (self.pixel_size_rad * self.N) / 2.0 - self.pixel_size_rad / 2.0
        ylist = np.arange(0, -self.N, -1) * self.pixel_size_rad + \
                (self.pixel_size_rad * self.N) / 2.0 - self.pixel_size_rad / 2.0

        # Create meshgrid and flatten
        xx, yy = np.meshgrid(xlist, ylist)
        x_vec = xx.flatten()
        y_vec = yy.flatten()

        # Build DFT matrix
        A = np.zeros((M, npixels), dtype=complex)
        for i, (u, v) in enumerate(self.uv_coords):
            A[i, :] = np.exp(-1j * 2.0 * np.pi * (u * x_vec + v * y_vec))

        return A

    def forward(self, image_vec):
        """Apply forward model: visibility = A @ image."""
        return self._A @ image_vec

    def adjoint(self, vis_vec):
        """Apply adjoint: image = A^H @ visibility."""
        return self._A.conj().T @ vis_vec

    @property
    def matrix(self):
        """Return the DFT matrix."""
        return self._A


# ============================================================================
# Measurement functions (lines 822-927 from starwarps.py)
# ============================================================================

def compute_visibilities(imvec, A):
    """Compute complex visibilities.

    Args:
        imvec: (N²,) image vector
        A: DFT matrix (M, N²)

    Returns:
        (M,) complex visibility vector
    """
    return np.dot(A, imvec)


def compute_bispectrum(imvec, A_matrices):
    """Compute bispectrum (product of 3 visibilities on triangle).

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 3 DFT matrices for the 3 baselines

    Returns:
        complex bispectrum values
    """
    return np.dot(A_matrices[0], imvec) * np.dot(A_matrices[1], imvec) * np.dot(A_matrices[2], imvec)


def compute_closure_phase(imvec, A_matrices):
    """Compute closure phase as complex exponential e^{i*cphase}.

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 3 DFT matrices for the 3 baselines

    Returns:
        complex exponential of closure phase
    """
    i1 = np.dot(A_matrices[0], imvec)
    i2 = np.dot(A_matrices[1], imvec)
    i3 = np.dot(A_matrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)
    return np.exp(1j * clphase_samples)


def compute_visibility_amplitude(imvec, A):
    """Compute visibility amplitude.

    Args:
        imvec: (N²,) image vector
        A: DFT matrix (M, N²)

    Returns:
        real amplitude values
    """
    i1 = np.dot(A, imvec)
    return np.abs(i1)


def compute_log_closure_amplitude(imvec, A_matrices):
    """Compute log closure amplitude.

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 4 DFT matrices for the 4 baselines

    Returns:
        real log closure amplitude values
    """
    i1 = np.dot(A_matrices[0], imvec)
    i2 = np.dot(A_matrices[1], imvec)
    i3 = np.dot(A_matrices[2], imvec)
    i4 = np.dot(A_matrices[3], imvec)
    return np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))


# ============================================================================
# Gradient functions (lines 827-927 from starwarps.py)
# ============================================================================

def grad_vis(imvec, A):
    """Gradient of visibility measurement (just the DFT matrix itself).

    Args:
        imvec: (N²,) image vector
        A: DFT matrix (M, N²)

    Returns:
        (M, N²) gradient matrix
    """
    return A


def grad_bispectrum(imvec, A_matrices):
    """Gradient of bispectrum.

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 3 DFT matrices

    Returns:
        (M, N²) gradient matrix
    """
    pt1 = np.dot(A_matrices[1], imvec) * np.dot(A_matrices[2], imvec)
    pt2 = np.dot(A_matrices[0], imvec) * np.dot(A_matrices[2], imvec)
    pt3 = np.dot(A_matrices[0], imvec) * np.dot(A_matrices[1], imvec)
    out = pt1[:, None] * A_matrices[0] + pt2[:, None] * A_matrices[1] + pt3[:, None] * A_matrices[2]
    return out


def grad_closure_phase(imvec, A_matrices):
    """Gradient of closure phase.

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 3 DFT matrices

    Returns:
        (M, N²) gradient matrix
    """
    i1 = np.dot(A_matrices[0], imvec)
    i2 = np.dot(A_matrices[1], imvec)
    i3 = np.dot(A_matrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)
    pt1 = 1.0 / i1
    pt2 = 1.0 / i2
    pt3 = 1.0 / i3
    dphi = (pt1[:, None] * A_matrices[0]) + (pt2[:, None] * A_matrices[1]) + (pt3[:, None] * A_matrices[2])
    out = 1j * np.imag(dphi) * np.exp(1j * clphase_samples[:, None])
    return out


def grad_visibility_amplitude(imvec, A):
    """Gradient of visibility amplitude.

    Args:
        imvec: (N²,) image vector
        A: DFT matrix (M, N²)

    Returns:
        (M, N²) gradient matrix
    """
    i1 = np.dot(A, imvec)
    pp = np.abs(i1) / i1
    out = np.real(pp[:, None] * A)
    return out


def grad_log_closure_amplitude(imvec, A_matrices):
    """Gradient of log closure amplitude.

    Args:
        imvec: (N²,) image vector
        A_matrices: list of 4 DFT matrices

    Returns:
        (M, N²) gradient matrix
    """
    i1 = np.dot(A_matrices[0], imvec)
    i2 = np.dot(A_matrices[1], imvec)
    i3 = np.dot(A_matrices[2], imvec)
    i4 = np.dot(A_matrices[3], imvec)

    pt1 = 1 / i1
    pt2 = 1 / i2
    pt3 = -1 / i3
    pt4 = -1 / i4
    out = np.real(pt1[:, None] * A_matrices[0] + pt2[:, None] * A_matrices[1] +
                  pt3[:, None] * A_matrices[2] + pt4[:, None] * A_matrices[3])
    return out


def compute_flux(imvec):
    """Compute total flux (sum of pixel intensities).

    Args:
        imvec: (N²,) image vector

    Returns:
        scalar total flux
    """
    return np.sum(imvec)


def grad_flux(imvec):
    """Gradient of flux (ones vector).

    Args:
        imvec: (N²,) image vector

    Returns:
        (1, N²) gradient matrix
    """
    return np.ones((1, len(imvec)))


# ============================================================================
# Helper functions (lines 1490-1576 from starwarps.py)
# ============================================================================

def gen_freq_comp(N, psize):
    """Generate frequency components for an N x N image.

    Adapted from starwarps.py lines 1490-1503. Original used im.xdim/im.ydim;
    here we take N and psize directly.

    Args:
        N: image dimension (assumes square NxN)
        psize: pixel size in radians

    Returns:
        (ufull, vfull): each (N², 1) arrays of frequency coordinates
    """
    fN2 = int(np.floor(N / 2))
    fM2 = int(np.floor(N / 2))

    ulist = (np.array([np.concatenate((
        np.linspace(0, fN2 - 1, fN2),
        np.linspace(-fN2, -1, fN2)
    ), axis=0)]) / N) / psize

    vlist = (np.array([np.concatenate((
        np.linspace(0, fM2 - 1, fM2),
        np.linspace(-fM2, -1, fM2)
    ), axis=0)]) / N) / psize

    ufull, vfull = np.meshgrid(ulist, vlist)

    ufull = np.reshape(ufull, (N * N, -1), order='F')
    vfull = np.reshape(vfull, (N * N, -1), order='F')

    return (ufull, vfull)


def gen_phase_shift_matrix(ulist, vlist, init_x, init_y, flowbasis_x, flowbasis_y,
                           theta, psize):
    """Generate phase shift matrix (DFT-based warp).

    Adapted from starwarps.py lines 1315-1333.
    Uses delta pulse (always 1.0) instead of ehtim.observing.pulses.deltaPulse2D.

    Args:
        ulist: (Nfreq, 1) array of u frequencies
        vlist: (Nfreq, 1) array of v frequencies
        init_x, init_y: initial position arrays (N, N, 1)
        flowbasis_x, flowbasis_y: flow basis arrays (N, N, nbasis)
        theta: (nbasis,) parameter vector
        psize: pixel size in radians

    Returns:
        (Nfreq, Npixels) complex phase shift matrix
    """
    flow_x, flow_y = apply_motion_basis(init_x, init_y, flowbasis_x, flowbasis_y, theta)

    imsize = flow_x.shape
    npixels = np.prod(imsize)

    flow_x_vec = np.reshape(flow_x, (npixels))
    flow_y_vec = np.reshape(flow_y, (npixels))

    shiftMtx_y = np.exp([-1j * 2.0 * np.pi * flow_y_vec * v for v in vlist])
    shiftMtx_x = np.exp([-1j * 2.0 * np.pi * flow_x_vec * u for u in ulist])

    uvlist = np.transpose(np.squeeze(np.array([ulist, vlist])))
    uvlist = np.reshape(uvlist, (vlist.shape[0], 2))
    # Delta pulse: always returns 1.0
    pulseVec = [delta_pulse_2d(2 * np.pi * uv[0], 2 * np.pi * uv[1], psize, dom="F")
                for uv in uvlist]

    shiftMtx = np.dot(np.diag(pulseVec),
                       np.reshape(np.squeeze(shiftMtx_x * shiftMtx_y),
                                  (vlist.shape[0], npixels)))
    return shiftMtx


def apply_motion_basis(init_x, init_y, flowbasis_x, flowbasis_y, theta):
    """Apply motion basis to compute flow fields.

    Adapted from starwarps.py lines 1208-1217.

    Args:
        init_x, init_y: initial position arrays (N, N, 1)
        flowbasis_x, flowbasis_y: flow basis arrays (N, N, nbasis)
        theta: (nbasis,) parameter vector

    Returns:
        (flow_x, flow_y): each (N, N) arrays of flow field
    """
    imsize = flowbasis_x.shape[0:2]
    nbasis = theta.shape[0]
    npixels = np.prod(imsize)

    flow_x = init_x[:, :, 0] + np.reshape(
        np.dot(np.reshape(flowbasis_x, (npixels, nbasis), order='F'), theta),
        imsize, order='F')
    flow_y = init_y[:, :, 0] + np.reshape(
        np.dot(np.reshape(flowbasis_y, (npixels, nbasis), order='F'), theta),
        imsize, order='F')

    return (flow_x, flow_y)


def realimag_stack(mtx):
    """Stack real and imaginary parts of a complex matrix vertically.

    Adapted from starwarps.py lines 1568-1570.

    Args:
        mtx: complex matrix

    Returns:
        vertically stacked [Re(mtx); Im(mtx)]
    """
    return np.concatenate((np.real(mtx), np.imag(mtx)), axis=0)


def reshape_flowbasis(flowbasis):
    """Reshape 3D flow basis to 2D.

    Adapted from starwarps.py lines 1572-1576.

    Args:
        flowbasis: (N, N, nbasis) array

    Returns:
        (N², nbasis) array
    """
    npixels = flowbasis.shape[0] * flowbasis.shape[1]
    return np.reshape(flowbasis, (npixels, -1))


# ============================================================================
# Gaussian prior covariance (lines 1077-1094 from starwarps.py)
# ============================================================================

def gauss_image_covariance(N, psize, imvec, power_dropoff=2.0, frac=0.5):
    """Construct Gaussian image covariance matrix with power-law spectrum.

    Adapted from gaussImgCovariance_2 in starwarps.py lines 1077-1094.
    Builds a full (N², N²) covariance matrix using the Fourier domain with
    power-law spectral weighting, modulated by the image intensities.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians
        imvec: (N²,) image vector (used to modulate covariance)
        power_dropoff: power-law exponent for UV distance weighting
        frac: scaling fraction

    Returns:
        (N², N²) covariance matrix
    """
    eps = 0.001

    init_x, init_y, flowbasis_x, flowbasis_y, init_theta = affine_motion_basis(N, psize)
    ufull, vfull = gen_freq_comp(N, psize)
    shiftMtx = gen_phase_shift_matrix(ufull, vfull, init_x, init_y,
                                      flowbasis_x, flowbasis_y, init_theta, psize)
    uvdist = np.reshape(np.sqrt(ufull**2 + vfull**2), (ufull.shape[0])) + eps
    uvdist = uvdist / np.min(uvdist)
    uvdist[0] = np.inf

    shiftMtx_exp = realimag_stack(shiftMtx)
    uvdist_exp = np.concatenate((uvdist, uvdist), axis=0)

    imCov = np.dot(np.transpose(shiftMtx_exp),
                   np.dot(np.diag(1 / (uvdist_exp**power_dropoff)), shiftMtx_exp))
    imCov = frac**2 * np.dot(np.diag(imvec).T,
                              np.dot(imCov / imCov[0, 0], np.diag(imvec)))
    return imCov


# ============================================================================
# Motion basis functions (lines 1123-1163 from starwarps.py)
# ============================================================================

def affine_motion_basis(N, psize):
    """Construct affine motion basis for an N x N image.

    Adapted from starwarps.py lines 1123-1135.
    Original took an ehtim Image object; this takes N and psize directly.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians

    Returns:
        (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)
        - init_x: (N, N, 1)
        - init_y: (N, N, 1)
        - flowbasis_x: (N, N, 6)
        - flowbasis_y: (N, N, 6)
        - init_theta: (6,) array [1, 0, 0, 0, 1, 0]
    """
    xlist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    ylist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0

    init_x = np.array([[[0] for i in xlist] for j in ylist])
    init_y = np.array([[[0] for i in xlist] for j in ylist])

    flowbasis_x = np.array([[[i, j, psize, 0, 0, 0] for i in xlist] for j in ylist])
    flowbasis_y = np.array([[[0, 0, 0, i, j, psize] for i in xlist] for j in ylist])
    init_theta = np.array([1, 0, 0, 0, 1, 0])

    return (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)


def affine_motion_basis_no_translation(N, psize):
    """Construct affine motion basis without translation terms.

    Adapted from starwarps.py lines 1137-1149.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians

    Returns:
        (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)
        - flowbasis_x: (N, N, 4)
        - flowbasis_y: (N, N, 4)
        - init_theta: (4,) array [1, 0, 0, 1]
    """
    xlist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    ylist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0

    init_x = np.array([[[0] for i in xlist] for j in ylist])
    init_y = np.array([[[0] for i in xlist] for j in ylist])

    flowbasis_x = np.array([[[i, j, 0, 0] for i in xlist] for j in ylist])
    flowbasis_y = np.array([[[0, 0, i, j] for i in xlist] for j in ylist])
    init_theta = np.array([1, 0, 0, 1])

    return (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)


def translation_basis(N, psize):
    """Construct translation-only motion basis.

    Adapted from starwarps.py lines 1151-1163.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians

    Returns:
        (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)
        - init_x: (N, N, 1) with pixel x positions
        - init_y: (N, N, 1) with pixel y positions
        - flowbasis_x: (N, N, 2)
        - flowbasis_y: (N, N, 2)
        - init_theta: (2,) array [0.0, 0.0]
    """
    xlist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    ylist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0

    init_x = np.array([[[i] for i in xlist] for j in ylist])
    init_y = np.array([[[j] for i in xlist] for j in ylist])

    flowbasis_x = np.array([[[psize, 0.0] for i in xlist] for j in ylist])
    flowbasis_y = np.array([[[0.0, psize] for i in xlist] for j in ylist])
    init_theta = np.array([0.0, 0.0])

    return (init_x, init_y, flowbasis_x, flowbasis_y, init_theta)


# ============================================================================
# Warp matrix (lines 1231-1258 from starwarps.py)
# ============================================================================

def calc_warp_matrix(N, psize, theta, init_x, init_y, flowbasis_x, flowbasis_y,
                     init_theta, method='phase'):
    """Calculate the warp matrix for image warping.

    Adapted from starwarps.py calcWarpMtx lines 1231-1258.
    Original used im.xdim, im.ydim, im.psize, im.pulse; here we pass N, psize directly.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians
        theta: (nbasis,) current warp parameters
        init_x, init_y: initial position arrays (N, N, 1)
        flowbasis_x, flowbasis_y: flow basis arrays (N, N, nbasis)
        init_theta: (nbasis,) identity/reference warp parameters
        method: 'phase' for Fourier-domain warping

    Returns:
        (N², N²) real warp matrix
    """
    npixels = N * N

    if method == 'phase':
        ufull, vfull = gen_freq_comp(N, psize)
        shiftMtx0 = gen_phase_shift_matrix(ufull, vfull, init_x, init_y,
                                           flowbasis_x, flowbasis_y, theta, psize)
        shiftMtx1 = gen_phase_shift_matrix(ufull, vfull, init_x, init_y,
                                           flowbasis_x, flowbasis_y, init_theta, psize)

        outMtx = np.real(np.dot(np.linalg.inv(shiftMtx1), shiftMtx0))

    elif method == 'img':
        outMtx = np.zeros((npixels, npixels))
        for i in range(npixels):
            probe_vec = np.zeros(npixels)
            probe_vec[i] = 1.0
            # For image-domain warping, would need interpolation (not implemented
            # since 'phase' is the standard method for StarWarps)
            raise NotImplementedError("Image-domain warping requires scipy.interpolate "
                                      "and is not the standard method for StarWarps.")

        outMtx = np.nan_to_num(outMtx)
    else:
        raise ValueError(f"Unknown method: {method}")

    return outMtx


# ============================================================================
# Warp gradient (lines 1337-1363 from starwarps.py)
# ============================================================================

def calc_dwarp_dtheta(N, psize, center_theta, init_x, init_y, flowbasis_x,
                      flowbasis_y, init_theta, method='phase'):
    """Calculate derivative of warp matrix w.r.t. theta parameters.

    Adapted from starwarps.py calc_dWarp_dTheta lines 1337-1363.

    Args:
        N: image dimension (N x N)
        psize: pixel size in radians
        center_theta: (nbasis,) center point for linearization
        init_x, init_y: initial position arrays (N, N, 1)
        flowbasis_x, flowbasis_y: flow basis arrays (N, N, nbasis)
        init_theta: (nbasis,) identity/reference warp parameters
        method: 'phase' for Fourier-domain warping

    Returns:
        list of nbasis (N², N²) matrices, one per basis element
    """
    if method == 'phase' or method == 'approx_phase':
        ufull, vfull = gen_freq_comp(N, psize)

        derivShiftMtx_x, derivShiftMtx_y = _calc_deriv_shift_mtx_freq(
            ufull, vfull, N, psize, center_theta, init_x, init_y,
            flowbasis_x, flowbasis_y, include_img_flow=False)

        shiftMtx1 = gen_phase_shift_matrix(ufull, vfull, init_x, init_y,
                                           flowbasis_x, flowbasis_y, init_theta, psize)
        invShiftMtx1 = np.linalg.inv(shiftMtx1)

        flowbasis = np.concatenate((reshape_flowbasis(flowbasis_x),
                                    reshape_flowbasis(flowbasis_y)), axis=0)

        reshape_fb_x = reshape_flowbasis(flowbasis_x)
        reshape_fb_y = reshape_flowbasis(flowbasis_y)

        dWarp_dTheta = []
        for b in range(flowbasis.shape[1]):
            K = (np.dot(derivShiftMtx_x, np.diag(reshape_fb_x[:, b])) +
                 np.dot(derivShiftMtx_y, np.diag(reshape_fb_y[:, b])))
            dWarp_dTheta.append(np.real(np.dot(invShiftMtx1, K)))

    else:
        raise NotImplementedError("Only 'phase' method is supported for warp gradient.")

    return dWarp_dTheta


def _calc_deriv_shift_mtx_freq(ulist, vlist, N, psize, center_theta,
                                init_x, init_y, flowbasis_x, flowbasis_y,
                                include_img_flow=False, imvec=None):
    """Calculate derivative of phase shift matrix in frequency domain.

    Adapted from starwarps.py calcDerivShiftMtx_freq lines 1418-1437.

    Args:
        ulist, vlist: frequency component arrays
        N: image dimension
        psize: pixel size in radians
        center_theta: center point for linearization
        init_x, init_y: initial position arrays
        flowbasis_x, flowbasis_y: flow basis arrays
        include_img_flow: if True, returns combined theta derivative matrix
        imvec: image vector (required if include_img_flow is True)

    Returns:
        If include_img_flow: single combined derivative matrix
        Else: (derivShiftMtx_x, derivShiftMtx_y) tuple
    """
    npixels = N * N
    shiftMtx = gen_phase_shift_matrix(ulist, vlist, init_x, init_y,
                                      flowbasis_x, flowbasis_y, center_theta, psize)

    shiftVec_y = np.array([-1j * 2.0 * np.pi * v * np.ones(npixels) for v in vlist])
    shiftVec_x = np.array([-1j * 2.0 * np.pi * u * np.ones(npixels) for u in ulist])
    derivShiftMtx_x = shiftVec_x * shiftMtx
    derivShiftMtx_y = shiftVec_y * shiftMtx

    if include_img_flow:
        if imvec is None:
            raise ValueError("imvec required when include_img_flow is True")
        flowbasis = np.concatenate((reshape_flowbasis(flowbasis_x),
                                    reshape_flowbasis(flowbasis_y)), axis=0)
        derivShiftMtx = np.concatenate(
            (np.dot(derivShiftMtx_x, np.diag(imvec)),
             np.dot(derivShiftMtx_y, np.diag(imvec))),
            axis=1)
        thetaDerivShiftMtx = np.dot(derivShiftMtx, flowbasis)
        return thetaDerivShiftMtx
    else:
        return (derivShiftMtx_x, derivShiftMtx_y)


# ============================================================================
# Measurement term assembly (lines 744-820 from starwarps.py)
# ============================================================================

def get_measurement_terms(dft_model, imvec, vis_data, sigma_data, measurement='vis',
                          A_matrices=None, weight=1.0, normalize=False):
    """Assemble Jacobian, measurement residual, and covariance for StarWarps update.

    Adapted from starwarps.py getMeasurementTerms lines 744-820.
    Replaces chisqdata() call with DFTForwardModel.matrix.

    Args:
        dft_model: DFTForwardModel instance
        imvec: (N²,) image vector (linearization point)
        vis_data: measured data array (complex visibilities, etc.)
        sigma_data: measurement uncertainties
        measurement: one of 'vis', 'bs', 'cphase', 'amp', 'logcamp', 'flux'
        A_matrices: for 'bs'/'cphase', list of 3 DFT matrices;
                    for 'logcamp', list of 4 DFT matrices.
                    For 'vis' and 'amp', uses dft_model.matrix.
        weight: weighting factor for this data term
        normalize: whether to normalize sigma

    Returns:
        (meas_diff, ideal_meas, jacobian_F, meas_cov, valid)
        - meas_diff: measured - ideal + F @ imvec (linearized residual)
        - ideal_meas: model-predicted measurements
        - jacobian_F: derivative matrix
        - meas_cov: measurement noise covariance (diagonal)
        - valid: bool
    """
    A = dft_model.matrix
    data = vis_data
    sigma = sigma_data

    # Compute the derivative matrix and ideal measurements
    if measurement == 'vis':
        F = A
        ideal = compute_visibilities(imvec, A)
    elif measurement == 'bs':
        if A_matrices is None:
            raise ValueError("A_matrices required for bispectrum")
        F = grad_bispectrum(imvec, A_matrices)
        ideal = compute_bispectrum(imvec, A_matrices)
    elif measurement == 'cphase':
        if A_matrices is None:
            raise ValueError("A_matrices required for closure phase")
        F = grad_closure_phase(imvec, A_matrices)
        ideal = compute_closure_phase(imvec, A_matrices)
    elif measurement == 'amp':
        F = grad_visibility_amplitude(imvec, A)
        ideal = compute_visibility_amplitude(imvec, A)
    elif measurement == 'logcamp':
        if A_matrices is None:
            raise ValueError("A_matrices required for log closure amplitude")
        F = grad_log_closure_amplitude(imvec, A_matrices)
        ideal = compute_log_closure_amplitude(imvec, A_matrices)
    elif measurement == 'flux':
        F = grad_flux(imvec)
        ideal = compute_flux(imvec)
    else:
        return (-1, -1, -1, -1, False)

    # Turn complex matrices to real
    if not np.allclose(np.imag(data), 0):
        F = realimag_stack(F)
        data = realimag_stack(data)
        ideal = realimag_stack(ideal)
        sigma = np.concatenate((sigma, sigma), axis=0)

    # Apply weight and normalization
    if normalize:
        sigma = sigma / np.sqrt(np.sum(sigma ** 2))
    sigma = sigma / np.sqrt(weight)
    Cov = np.diag(sigma ** 2)

    # Linearized measurement residual: data + F @ imvec - ideal
    meas_diff = data.reshape(-1) + np.dot(F, imvec) - ideal.reshape(-1)

    return (meas_diff, ideal.reshape(-1), F, Cov, True)


def get_measurement_terms_multi(dft_model, imvec, measurements_dict, data_dict,
                                sigma_dict, A_matrices_dict=None, normalize=False):
    """Assemble measurement terms for multiple data products.

    This mirrors the loop in the original getMeasurementTerms that iterates
    over multiple measurement types (e.g., {'vis': 1, 'cphase': 0.5}).

    Args:
        dft_model: DFTForwardModel instance
        imvec: (N²,) image vector
        measurements_dict: dict mapping measurement name -> weight
        data_dict: dict mapping measurement name -> data array
        sigma_dict: dict mapping measurement name -> sigma array
        A_matrices_dict: dict mapping measurement name -> A_matrices (for bs/cphase/logcamp)
        normalize: whether to normalize sigma

    Returns:
        (meas_diff, ideal_meas, jacobian_F, meas_cov, valid)
    """
    measdiff_all = []
    ideal_all = []
    F_all = []
    Cov_all = []
    count = 0

    for dname, weight in measurements_dict.items():
        if np.allclose(weight, 0.0):
            continue

        if dname not in data_dict:
            continue

        A_mats = None
        if A_matrices_dict is not None and dname in A_matrices_dict:
            A_mats = A_matrices_dict[dname]

        try:
            meas_diff, ideal, F, Cov, valid = get_measurement_terms(
                dft_model, imvec, data_dict[dname], sigma_dict[dname],
                measurement=dname, A_matrices=A_mats, weight=weight,
                normalize=normalize)
            if not valid:
                continue
        except Exception:
            continue

        count += 1
        measdiff_all = np.concatenate((measdiff_all, meas_diff.reshape(-1)), axis=0).reshape(-1)
        ideal_all = np.concatenate((ideal_all, ideal.reshape(-1)), axis=0).reshape(-1)
        F_all = np.concatenate((F_all, F), axis=0) if len(F_all) else F
        Cov_all = scipy.linalg.block_diag(Cov_all, Cov) if len(Cov_all) else Cov

    if len(measdiff_all):
        return (measdiff_all, ideal_all, F_all, Cov_all, True)
    else:
        return (-1, -1, -1, -1, False)


# ============================================================================
# EM M-step functions (lines 609-674 from starwarps.py)
# ============================================================================

def exp_neg_loglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t,
                          N, psize, Upsilon, init_x, init_y,
                          flowbasis_x, flowbasis_y, init_theta, method='phase'):
    """Expected negative log-likelihood for the M-step.

    Adapted from starwarps.py expnegloglikelihood lines 609-636.
    This is the objective function for the M-step optimizer.

    The original used mu[0] (an ehtim Image) to build the warp matrix;
    here we use N, psize directly.

    Args:
        theta: (nbasis,) current warp parameters to optimize
        expVal_t: list of (N²,) expected image vectors E[x_t]
        expVal_t_t: list of (N², N²) expected E[x_t x_t^T] matrices
        expVal_tm1_t: list of (N², N²) expected E[x_{t-1} x_t^T] matrices
        N: image dimension
        psize: pixel size in radians
        Upsilon: (N², N²) process noise covariance Q
        init_x, init_y: initial position arrays
        flowbasis_x, flowbasis_y: flow basis arrays
        init_theta: reference warp parameters
        method: warp method ('phase')

    Returns:
        scalar expected negative log-likelihood value
    """
    warpMtx = calc_warp_matrix(N, psize, theta, init_x, init_y,
                                flowbasis_x, flowbasis_y, init_theta, method=method)
    A = warpMtx
    Q = Upsilon
    invQ = np.linalg.inv(Q)

    value = 0.0
    for t in range(1, len(expVal_t)):
        x_t = np.array([expVal_t[t]]).T
        x_tm1 = np.array([expVal_t[t - 1]]).T

        P_tm1_t = expVal_tm1_t[t] - np.dot(x_tm1, x_t.T)
        P_tm1_tm1 = expVal_t_t[t - 1] - np.dot(x_tm1, x_tm1.T)

        term1 = _exp_xtm1_M_xt(P_tm1_t.T, x_t, x_tm1, np.dot(invQ, A))
        term2 = _exp_xtm1_M_xt(P_tm1_t, x_tm1, x_t, np.dot(A.T, invQ))
        term3 = _exp_xtm1_M_xt(P_tm1_tm1, x_tm1, x_tm1, np.dot(A.T, np.dot(invQ, A)))

        value = value - 0.5 * (-term1 - term2 + term3)

    value = -value
    return value


def deriv_exp_neg_loglikelihood(theta, expVal_t, expVal_t_t, expVal_tm1_t,
                                N, psize, Upsilon, init_x, init_y,
                                flowbasis_x, flowbasis_y, init_theta, method='phase'):
    """Gradient of expected negative log-likelihood for the M-step.

    Adapted from starwarps.py deriv_expnegloglikelihood lines 650-674.

    Args:
        theta: (nbasis,) current warp parameters
        expVal_t: list of (N²,) expected image vectors
        expVal_t_t: list of (N², N²) expected E[x_t x_t^T] matrices
        expVal_tm1_t: list of (N², N²) expected E[x_{t-1} x_t^T] matrices
        N: image dimension
        psize: pixel size in radians
        Upsilon: (N², N²) process noise covariance Q
        init_x, init_y: initial position arrays
        flowbasis_x, flowbasis_y: flow basis arrays
        init_theta: reference warp parameters
        method: warp method ('phase')

    Returns:
        (nbasis,) gradient vector
    """
    if method == 'phase':
        dWarp_dTheta = calc_dwarp_dtheta(N, psize, theta, init_x, init_y,
                                         flowbasis_x, flowbasis_y, init_theta,
                                         method=method)
    else:
        raise NotImplementedError("Only 'phase' method supported for warp gradient.")

    warpMtx = calc_warp_matrix(N, psize, theta, init_x, init_y,
                                flowbasis_x, flowbasis_y, init_theta, method=method)

    invQ = np.linalg.inv(Upsilon)
    M1 = np.zeros(expVal_tm1_t[1].shape)
    for t in range(1, len(expVal_t_t)):
        M1 = M1 + expVal_tm1_t[t].T - np.dot(warpMtx, expVal_t_t[t - 1])
    M1 = np.dot(invQ, M1)

    deriv = np.zeros(init_theta.shape)
    for b in range(len(init_theta)):
        for p in range(dWarp_dTheta[b].shape[0]):
            for q in range(dWarp_dTheta[b].shape[1]):
                deriv[b] = deriv[b] + M1[p, q] * dWarp_dTheta[b][p, q]

    # The derivative computed is for the ll but we want the derivative of the neg ll
    deriv = -deriv
    return deriv


def _exp_xtm1_M_xt(P, z1, z2, M):
    """Helper: compute E[x1^T M x2] = trace(P M^T) + z1^T M z2.

    Adapted from starwarps.py exp_xtm1_M_xt line 638-640.
    """
    value = np.trace(np.dot(P, M.T)) + np.dot(z1.T, np.dot(M, z2))
    return value


# ============================================================================
# Gaussian algebra (from starwarps.py lines 706-741)
# ============================================================================

def evaluate_gaussian_log(y, x, Sigma):
    """Evaluate log probability of Gaussian distribution.

    log N(y; x, Sigma)

    Args:
        y: observed value vector
        x: mean vector
        Sigma: covariance matrix

    Returns:
        scalar log probability
    """
    n = len(x)
    diff = x - y
    (sign, logdet) = np.linalg.slogdet(Sigma)
    expval_log = (-(n / 2.0) * np.log(2.0 * np.pi)
                  - 0.5 * (sign * logdet)
                  - 0.5 * np.dot(diff.T, np.dot(np.linalg.inv(Sigma), diff)))
    return expval_log


def product_gaussians_lem1(m1, S1, m2, S2):
    """Product of two Gaussians (Lemma 1 from StarWarps supplementary).

    N(m1, S1) * N(m2, S2) proportional to N(mean, covariance)

    Args:
        m1, m2: mean vectors
        S1, S2: covariance matrices

    Returns:
        (mean, covariance)
    """
    K = np.linalg.inv(S1 + S2)
    covariance = np.dot(S1, np.dot(K, S2))
    mean = np.dot(S1, np.dot(K, m2)) + np.dot(S2, np.dot(K, m1))
    return (mean, covariance)


def product_gaussians_lem2(A, Sigma, y, mu, Q):
    """Product of Gaussian with linear observation (Lemma 2).

    Posterior of x given y = Ax + noise(Sigma), prior x ~ N(mu, Q)

    Args:
        A: measurement matrix (or warp matrix)
        Sigma: measurement noise covariance
        y: observation
        mu: prior mean
        Q: prior covariance

    Returns:
        (mean, covariance)
    """
    K1 = np.linalg.inv(Sigma + np.dot(A, np.dot(Q, np.transpose(A))))
    K2 = np.dot(Q, np.dot(A.T, K1))
    covariance = Q - np.dot(K2, np.dot(A, Q))
    mean = mu + np.dot(K2, y - np.dot(A, mu))
    return (mean, covariance)
