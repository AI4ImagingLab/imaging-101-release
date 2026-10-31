"""
ADMM solver for lensless (DiffuserCam) image reconstruction.

Problem
-------
Recover an RGB image v ∈ ℝ^{H×W×3} from a lensless sensor measurement
b ∈ ℝ^{H×W×3} given the calibration PSF h ∈ ℝ^{H×W×3}:

    min_{v ≥ 0}  (1/2)||b - Av||²₂  +  τ ||Ψv||₁

where  A v = crop(h * v)  and  Ψv  is the image gradient (finite differences).

Variable splitting
------------------
Following Biscarrat et al. (2018) and LenslessPiCam (Bezzam et al., 2023),
we introduce auxiliary variables and optimise entirely in the **padded** space:

    min_{v_pad, u, x, w}  (1/2)||b - Cx||²₂ + τ||u||₁ + 𝟙₊(w)
    s.t.  x = M v_pad,  u = Ψ v_pad,  w = v_pad

where M is the full (uncropped) padded circular convolution, C is the crop
operator, and v_pad is v placed at [start:end] in the padded array.

ADMM update rules (k → k+1)
-----------------------------
    u_{k+1} = S_{τ/μ₂}(Ψ v_pad,k + η_k/μ₂)                [soft threshold]
    x_{k+1} = X_divmat · (ξ_k + μ₁ M v_pad,k + pad(b))     [element-wise]
    w_{k+1} = max(ρ_k/μ₃ + v_pad,k, 0)                     [nonneg proj]
    v_pad,k+1 = ifft(R_divmat · fft(r_k))                   [freq domain]
    ξ_{k+1} = ξ_k + μ₁(M v_pad,k+1 - x_{k+1})
    η_{k+1} = η_k + μ₂(Ψ v_pad,k+1 - u_{k+1})
    ρ_{k+1} = ρ_k + μ₃(v_pad,k+1 - w_{k+1})

where:
    r_k = (μ₃ w - ρ_k) + Ψᵀ(μ₂ u - η_k) + M^H(μ₁ x - ξ_k)
    R_divmat  = 1 / (μ₁|H̃|² + μ₂|Ψ̃|² + μ₃)    [precomputed, freq domain]
    X_divmat  = 1 / (pad(ones) + μ₁)              [precomputed, padded space]
                = 1/(1+μ₁) in [start:end],  1/μ₁ outside

References
----------
Biscarrat et al. (2018). Build your own DiffuserCam. Tutorial.
Boyd et al. (2011). ADMM. Found. Trends Mach. Learn., 3(1).
Bezzam et al. (2023). LenslessPiCam. JOSS, 8(86).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft2, irfft2
from tqdm import tqdm

from .physics_model import RealFFTConvolve2D


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def soft_thresh(x: np.ndarray, thresh: float) -> np.ndarray:
    """Element-wise soft-thresholding: sign(x) · max(|x| − thresh, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


def finite_diff(v: np.ndarray) -> np.ndarray:
    """2-D finite-difference gradient operator Ψ.

    Computes row-wise and column-wise first differences with periodic BC.

    Parameters
    ----------
    v : ndarray, shape (H, W, C)   (or padded shape (pH, pW, C))

    Returns
    -------
    u : ndarray, shape (2, H, W, C)   [axis-0 indexes direction]
    """
    return np.stack([
        np.roll(v, 1, axis=0) - v,   # row (vertical) difference
        np.roll(v, 1, axis=1) - v,   # col (horizontal) difference
    ], axis=0)


def finite_diff_adj(u: np.ndarray) -> np.ndarray:
    """Adjoint of Ψ: Ψᵀu.

    Since Ψv = roll(v, 1) - v, the adjoint satisfies <Ψv, u> = <v, Ψᵀu>:
        Ψᵀu[i] = u[i+1] - u[i] = roll(u, -1) - u

    Parameters
    ----------
    u : ndarray, shape (2, H, W, C)

    Returns
    -------
    v : ndarray, shape (H, W, C)
    """
    return (
        (np.roll(u[0], -1, axis=0) - u[0])
        + (np.roll(u[1], -1, axis=1) - u[1])
    )


def finite_diff_gram(shape: tuple, dtype=np.float32) -> np.ndarray:
    """Frequency-domain Gram matrix of the 2-D finite-difference operator.

    ΨᵀΨ is diagonalised by the DFT.  Its eigenvalues are the DFT of the
    discrete Laplacian kernel (for the roll-based periodic BC used above):

        gram[0,0] = 4,  gram[0,1] = gram[0,-1] = gram[1,0] = gram[-1,0] = -1

    Parameters
    ----------
    shape : (H, W)   spatial dimensions (may be padded).
    dtype : numpy dtype

    Returns
    -------
    gram_fft : ndarray, shape (H, W//2+1, 1) complex — ready to broadcast.
    """
    H, W = shape
    gram = np.zeros((H, W), dtype=dtype)
    gram[0, 0]  =  4
    gram[0, 1]  = -1
    gram[0, -1] = -1
    gram[1, 0]  = -1
    gram[-1, 0] = -1
    return rfft2(gram)[:, :, np.newaxis]    # (H, W//2+1, 1)


# ---------------------------------------------------------------------------
# ADMM solver
# ---------------------------------------------------------------------------

class ADMM:
    """ADMM solver for lensless imaging with TV + non-negativity prior.

    The optimisation variable lives entirely in the **padded** space
    (shape (pH, pW, C)).  The final image is obtained by cropping.

    Parameters
    ----------
    psf : ndarray, shape (H, W, C)
        Normalised PSF (peak value 1, background subtracted).
    mu1, mu2, mu3 : float
        Augmented-Lagrangian penalty parameters.
    tau : float
        TV (L1) regularisation weight.
    """

    def __init__(
        self,
        psf: np.ndarray,
        mu1: float = 1e-6,
        mu2: float = 1e-5,
        mu3: float = 4e-5,
        tau: float = 1e-4,
    ):
        self._psf   = psf
        self._shape = psf.shape     # (H, W, C)
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.tau = tau

        # Build the convolver (norm="backward" matches LenslessPiCam default)
        self._conv = RealFFTConvolve2D(psf, norm="backward")

        self._measurement = None   # set by set_data()

        # All primal/dual variables are in padded space
        self._image_est = None   # v_pad  (pH, pW, C)
        self._u   = None         # u = Ψ v_pad  (2, pH, pW, C)
        self._x   = None         # x = M v_pad  (pH, pW, C)
        self._w   = None         # w = v_pad    (pH, pW, C)
        self._xi  = None         # dual for x = Mv  (pH, pW, C)
        self._eta = None         # dual for u = Ψv  (2, pH, pW, C)
        self._rho = None         # dual for w = v   (pH, pW, C)

        # Precomputed (set in set_data / reset)
        self._forward_out = None    # M v (cached to avoid recompute)
        self._Psi_out     = None    # Ψ v (cached)
        self._R_divmat    = None
        self._X_divmat    = None
        self._data_padded = None    # pad(b)

    def set_data(self, measurement: np.ndarray):
        """Set lensless measurement and precompute derived quantities.

        Parameters
        ----------
        measurement : ndarray, shape (H, W, C), float32, values in [0, 1]
        """
        H, W, C = self._shape
        assert measurement.shape == (H, W, C), (
            f"Measurement {measurement.shape} != PSF {(H, W, C)}"
        )
        self._measurement = measurement

        ph, pw = self._conv.padded_shape

        # pad(b): center b in padded array at [start:end]
        self._data_padded = self._conv._pad(measurement)

        # X_divmat = 1 / (pad(ones) + mu1)
        # pad(ones) = 1 at [start:end], 0 elsewhere
        ones_padded = self._conv._pad(np.ones((H, W, C), dtype=np.float32))
        self._X_divmat = (1.0 / (ones_padded + self.mu1)).astype(np.float32)

        # R_divmat = 1 / (mu1 |H|² + mu2 |Ψ̃|² + mu3)
        # Computed in float64: R_divmat.max() ≈ 1/mu3 ≈ 25000, so float32 (~1e-7
        # relative error) introduces ~2.5e-3 absolute error per iteration that
        # compounds across iterations and causes divergence.
        # |H|²: from rfft2(center_pad(psf)), shape (pH, pW//2+1, C)
        HtH = np.abs(self._conv.Hadj.astype(np.complex128) * self._conv.H.astype(np.complex128))

        # |Ψ̃|²: from finite_diff_gram on padded shape
        PsiTPsi = np.abs(finite_diff_gram((ph, pw), dtype=np.float64))  # (pH, pW//2+1, 1)

        self._R_divmat = 1.0 / (
            self.mu1 * HtH + self.mu2 * PsiTPsi + self.mu3
        )
        # R_divmat is float64; used in float64 v-update for numerical stability

        self.reset()

    def reset(self):
        """Initialise all primal and dual variables to zero."""
        ph, pw = self._conv.padded_shape
        C = self._shape[2]
        # Primal
        self._image_est = np.zeros((ph, pw, C), dtype=np.float32)
        self._u  = np.zeros((2, ph, pw, C), dtype=np.float32)
        self._x  = np.zeros((ph, pw, C), dtype=np.float32)
        self._w  = np.zeros((ph, pw, C), dtype=np.float32)
        # Dual
        self._xi  = np.zeros((ph, pw, C), dtype=np.float32)
        self._eta = np.zeros((2, ph, pw, C), dtype=np.float32)
        self._rho = np.zeros((ph, pw, C), dtype=np.float32)
        # Cached operators
        self._forward_out = self._conv.convolve(self._image_est)
        self._Psi_out     = finite_diff(self._image_est)

    # ------------------------------------------------------------------
    # ADMM sub-updates
    # ------------------------------------------------------------------

    def _U_update(self):
        """u ← S_{τ/μ₂}(Ψ v_pad + η/μ₂)"""
        self._u = soft_thresh(
            self._Psi_out + self._eta / self.mu2,
            self.tau / self.mu2,
        )

    def _X_update(self):
        """x ← X_divmat · (ξ + μ₁ M v_pad + pad(b))"""
        self._x = self._X_divmat * (
            self._xi + self.mu1 * self._forward_out + self._data_padded
        )

    def _W_update(self):
        """w ← max(ρ/μ₃ + v_pad, 0)"""
        self._w = np.maximum(self._rho / self.mu3 + self._image_est, 0.0)

    def _image_update(self):
        """v_pad ← ifft(R_divmat · fft(r))

        r = (μ₃w - ρ) + Ψᵀ(μ₂u - η) + M^H(μ₁x - ξ)
        """
        rk = (
            (self.mu3 * self._w - self._rho)
            + finite_diff_adj(self.mu2 * self._u - self._eta)
            + self._conv.deconvolve(self.mu1 * self._x - self._xi)
        )
        # R_divmat is float64; float64 × complex64 = complex128 automatically.
        freq = self._R_divmat * rfft2(rk, axes=(0, 1), norm=self._conv.norm)
        self._image_est = irfft2(
            freq, axes=(0, 1),
            s=self._conv.padded_shape, norm=self._conv.norm
        ).real.astype(np.float32)

    def _xi_update(self):
        """ξ ← ξ + μ₁(M v_pad - x)"""
        self._xi += self.mu1 * (self._forward_out - self._x)

    def _eta_update(self):
        """η ← η + μ₂(Ψ v_pad - u)"""
        self._eta += self.mu2 * (self._Psi_out - self._u)

    def _rho_update(self):
        """ρ ← ρ + μ₃(v_pad - w)"""
        self._rho += self.mu3 * (self._image_est - self._w)

    def _update(self):
        """One full ADMM iteration."""
        self._U_update()
        self._X_update()
        self._W_update()
        self._image_update()

        # Refresh cached convolution and gradient for next iteration
        self._forward_out = self._conv.convolve(self._image_est)
        self._Psi_out     = finite_diff(self._image_est)

        self._xi_update()
        self._eta_update()
        self._rho_update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, n_iter: int = 100, verbose: bool = True) -> np.ndarray:
        """Run ADMM for n_iter iterations.

        Parameters
        ----------
        n_iter  : int
        verbose : bool   Show tqdm progress bar.

        Returns
        -------
        reconstruction : ndarray, shape (H, W, C), float32, values ≥ 0
        """
        assert self._measurement is not None, "Call set_data() first."
        it = tqdm(range(n_iter), desc="ADMM") if verbose else range(n_iter)
        for _ in it:
            self._update()
        return self.get_image()

    def get_image(self) -> np.ndarray:
        """Crop and clip the current padded estimate.

        Returns
        -------
        img : ndarray, shape (H, W, C), float32, non-negative
        """
        img = self._conv._crop(self._image_est)
        img = np.clip(img, 0, None)
        return img.astype(np.float32)
