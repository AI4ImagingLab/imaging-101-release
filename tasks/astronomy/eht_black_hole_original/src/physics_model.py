"""
Closure Forward Model for VLBI Imaging
========================================

Implements the DFT-based forward model and closure quantity chi-squared
terms following ehtim's conventions exactly:

- DFT sign: +2πi  (ehtim convention since Jan 2017)
- Pixel grid: xlist = arange(0, -N, -1)*psize + (psize*N)/2 - psize/2
- Pulse function: triangle pulse (ehtim default)
- Closure phase χ²: (2/N_CP) Σ (1 - cos(φ_obs - φ_model)) / σ²
- Log closure amp χ²: (1/N_CA) Σ ((logCA_obs - logCA_model) / σ)²
- Gradients: CP uses imag(), logCA uses real()

Reference
---------
Chael et al. (2018). ApJ 857, 23. Eqs. 11-12.
ehtim: https://github.com/achael/eht-imaging
"""

import math
import numpy as np


def _triangle_pulse_F(omega, pdim):
    """Fourier-domain triangle pulse response (matches ehtim)."""
    if omega == 0:
        return 1.0
    return (4.0 / (pdim**2 * omega**2)) * (math.sin((pdim * omega) / 2.0))**2


def _ftmatrix(psize, N, uv_coords):
    """
    Build DFT matrix matching ehtim's ftmatrix exactly.

    Sign convention: +2πi (agrees with BU data, ehtim since Jan 2017)
    Pixel grid: xlist = arange(0, -N, -1)*psize + (psize*N)/2 - psize/2

    Parameters
    ----------
    psize     : float — pixel size in radians
    N         : int — image size (N x N)
    uv_coords : (M, 2) — baseline (u, v) in wavelengths

    Returns
    -------
    A : (M, N²) complex DFT matrix
    """
    xlist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0
    ylist = np.arange(0, -N, -1) * psize + (psize * N) / 2.0 - psize / 2.0

    M = len(uv_coords)
    A = np.zeros((M, N * N), dtype=np.complex128)

    for m in range(M):
        u, v = uv_coords[m]
        # Triangle pulse (pixel response)
        pulse = (_triangle_pulse_F(2 * np.pi * u, psize) *
                 _triangle_pulse_F(2 * np.pi * v, psize))
        # Outer product: exp(+2πi * y * v) ⊗ exp(+2πi * x * u)
        row = pulse * np.outer(
            np.exp(2j * np.pi * ylist * v),
            np.exp(2j * np.pi * xlist * u)
        )
        A[m] = row.ravel()

    return A


class ClosureForwardModel:
    """
    VLBI forward model with closure quantity support.

    Parameters
    ----------
    uv_coords      : (M, 2) — baseline (u,v) coordinates in wavelengths
    N               : int — image size (N x N pixels)
    pixel_size_rad  : float — pixel size in radians
    triangles       : (N_tri, 3) int — station triples for closure phases
    quadrangles     : (N_quad, 4) int — station quads for closure amplitudes
    station_ids     : (M, 2) int — station pair indices (optional, for index-based closure)
    """

    def __init__(self, uv_coords, N, pixel_size_rad, triangles, quadrangles,
                 station_ids=None):
        self.N = N
        self.pixel_size_rad = pixel_size_rad
        self.uv_coords = uv_coords
        self.triangles = triangles
        self.quadrangles = quadrangles
        self.station_ids = station_ids

        # Build DFT matrix (matches ehtim ftmatrix)
        if len(uv_coords) > 0:
            self.A = _ftmatrix(pixel_size_rad, N, uv_coords)
        else:
            self.A = np.zeros((0, N * N), dtype=np.complex128)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """image (N,N) → complex visibilities (M,)"""
        return self.A @ image.ravel()

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """complex visibilities (M,) → back-projected image (N,N)"""
        return (self.A.conj().T @ vis).real.reshape(self.N, self.N)

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """Normalized back-projection (dirty image)."""
        M = len(vis)
        if M == 0:
            return np.zeros((self.N, self.N))
        raw = self.adjoint(vis)
        return raw / M

    def psf(self) -> np.ndarray:
        """Point spread function (dirty beam)."""
        M = self.A.shape[0]
        ones = np.ones(M, dtype=complex)
        return self.dirty_image(ones)

    # ── Visibility chi-squared ──────────────────────────────────────────

    def visibility_chisq(self, image: np.ndarray, vis_obs: np.ndarray,
                         sigma: np.ndarray) -> float:
        """
        Normalized visibility chi-squared.

        χ² = (1/M) Σ |V_model - V_obs|² / σ²
        """
        vis_model = self.forward(image)
        return float(np.sum(np.abs((vis_model - vis_obs) / sigma)**2) / len(vis_obs))

    def visibility_chisq_grad(self, image: np.ndarray, vis_obs: np.ndarray,
                              sigma: np.ndarray) -> np.ndarray:
        """Gradient of visibility chi-squared w.r.t. image."""
        vis_model = self.forward(image)
        residual = (vis_model - vis_obs) / sigma**2
        return (2.0 / len(vis_obs)) * (self.A.conj().T @ residual).real

    # ── Closure phase chi-squared (ehtim-compatible) ────────────────────

    @staticmethod
    def chisq_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3,
                              clphase_deg, sigma_deg):
        """
        Closure phase chi-squared matching ehtim exactly.

        χ² = (2/N_CP) Σ (1 - cos(φ_obs - φ_model)) / σ²

        Parameters
        ----------
        imvec       : (N²,) flat image in Jy/pixel
        N           : int image size
        psize       : float pixel size (rad)
        uv1, uv2, uv3 : (N_CP, 2) UV coords for each triangle leg
        clphase_deg : (N_CP,) observed closure phases in degrees
        sigma_deg   : (N_CP,) closure phase sigmas in degrees
        """
        DEGREE = np.pi / 180.0
        clphase = clphase_deg * DEGREE
        sigma = sigma_deg * DEGREE

        A1 = _ftmatrix(psize, N, uv1)
        A2 = _ftmatrix(psize, N, uv2)
        A3 = _ftmatrix(psize, N, uv3)

        i1 = A1 @ imvec
        i2 = A2 @ imvec
        i3 = A3 @ imvec
        clphase_model = np.angle(i1 * i2 * i3)

        chisq = (2.0 / len(clphase)) * np.sum(
            (1.0 - np.cos(clphase - clphase_model)) / sigma**2
        )
        return float(chisq)

    @staticmethod
    def chisqgrad_cphase_from_uv(imvec, N, psize, uv1, uv2, uv3,
                                  clphase_deg, sigma_deg):
        """
        Gradient of closure phase chi-squared (matches ehtim).

        ∂χ²/∂I = (-2/N_CP) Im[ Σ sin(φ_obs - φ_model)/σ² * (A_k^T / i_k) ]
        """
        DEGREE = np.pi / 180.0
        clphase = clphase_deg * DEGREE
        sigma = sigma_deg * DEGREE

        A1 = _ftmatrix(psize, N, uv1)
        A2 = _ftmatrix(psize, N, uv2)
        A3 = _ftmatrix(psize, N, uv3)

        i1 = A1 @ imvec
        i2 = A2 @ imvec
        i3 = A3 @ imvec
        clphase_model = np.angle(i1 * i2 * i3)

        pref = np.sin(clphase - clphase_model) / sigma**2
        pt1 = pref / i1
        pt2 = pref / i2
        pt3 = pref / i3

        out = pt1 @ A1 + pt2 @ A2 + pt3 @ A3
        out = (-2.0 / len(clphase)) * np.imag(out)
        return out

    # ── Log closure amplitude chi-squared (ehtim-compatible) ────────────

    @staticmethod
    def chisq_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4,
                               log_clamp, sigma):
        """
        Log closure amplitude chi-squared matching ehtim.

        χ² = (1/N_CA) Σ ((logCA_obs - logCA_model) / σ)²
        """
        A1 = _ftmatrix(psize, N, uv1)
        A2 = _ftmatrix(psize, N, uv2)
        A3 = _ftmatrix(psize, N, uv3)
        A4 = _ftmatrix(psize, N, uv4)

        a1 = np.abs(A1 @ imvec)
        a2 = np.abs(A2 @ imvec)
        a3 = np.abs(A3 @ imvec)
        a4 = np.abs(A4 @ imvec)

        samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)
        chisq = np.sum(np.abs((log_clamp - samples) / sigma)**2) / len(log_clamp)
        return float(chisq)

    @staticmethod
    def chisqgrad_logcamp_from_uv(imvec, N, psize, uv1, uv2, uv3, uv4,
                                   log_clamp, sigma):
        """
        Gradient of log closure amplitude chi-squared (matches ehtim).
        """
        A1 = _ftmatrix(psize, N, uv1)
        A2 = _ftmatrix(psize, N, uv2)
        A3 = _ftmatrix(psize, N, uv3)
        A4 = _ftmatrix(psize, N, uv4)

        i1 = A1 @ imvec
        i2 = A2 @ imvec
        i3 = A3 @ imvec
        i4 = A4 @ imvec

        log_clamp_model = (np.log(np.abs(i1)) + np.log(np.abs(i2))
                          - np.log(np.abs(i3)) - np.log(np.abs(i4)))

        pp = (log_clamp - log_clamp_model) / sigma**2
        pt1 = pp / i1
        pt2 = pp / i2
        pt3 = -pp / i3
        pt4 = -pp / i4

        out = pt1 @ A1 + pt2 @ A2 + pt3 @ A3 + pt4 @ A4
        out = (-2.0 / len(log_clamp)) * np.real(out)
        return out

    # ── Instance methods using pre-built A matrices ─────────────────────

    def model_closure_phases(self, image: np.ndarray) -> np.ndarray:
        """Compute model closure phases from image (radians)."""
        vis = self.forward(image)
        from src.preprocessing import compute_closure_phases
        return compute_closure_phases(vis, self.station_ids, self.triangles)

    def model_log_closure_amplitudes(self, image: np.ndarray) -> np.ndarray:
        """Compute model log closure amplitudes from image."""
        vis = self.forward(image)
        from src.preprocessing import compute_log_closure_amplitudes
        return compute_log_closure_amplitudes(vis, self.station_ids, self.quadrangles)

    def __repr__(self):
        return (f"ClosureForwardModel(M={self.A.shape[0]}, N={self.N}, "
                f"n_tri={len(self.triangles)}, n_quad={len(self.quadrangles)})")
