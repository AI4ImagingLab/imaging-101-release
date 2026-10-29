"""
Forward model for lensless (DiffuserCam) imaging.

Under the shift-invariant (LSI) approximation the lensless camera maps
a scene v to a sensor measurement b via a cropped 2-D convolution:

    b = crop( h * v )

where h is the measured point-spread function (PSF) and * denotes linear
(non-circular) 2-D convolution.  Because the sensor has finite extent,
only the central region of the full convolution result is recorded;
this is the "crop" operation.

Implementation convention
--------------------------
We follow the LenslessPiCam convention (Bezzam et al., 2023):

1. The PSF h is **center-padded** into a (pH, pW, C) array so that h sits
   at indices [start:end] = [(pH-H)//2 : (pH-H)//2+H, ...].  This
   centering is crucial for the ifftshift to work correctly.

2. After `irfft2`, we apply `scipy.fft.ifftshift` to move the zero-frequency
   component from the corner to the center of the padded output.  This
   accounts for the phase offset introduced by center-padding.

3. `crop` and `pad` use the same [start:end] window so that `crop(pad(v))=v`.

4. Inside the ADMM solver the image variable lives entirely in padded space
   (shape (pH, pW, C)).  Neither `convolve` nor `deconvolve` add extra
   padding; they operate directly on the padded array.

Reference
---------
Biscarrat et al. (2018). Build your own DiffuserCam. Tutorial.
Bezzam et al. (2023). LenslessPiCam. JOSS, 8(86).
"""

import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2, ifftshift


class RealFFTConvolve2D:
    """2-D convolution in the Fourier domain with a real-valued PSF.

    Parameters
    ----------
    psf : ndarray, shape (H, W, C)
        Normalised point-spread function.
    norm : str
        Normalisation for scipy.fft.rfft2 / irfft2 (default "backward",
        which applies no scaling on the forward transform and 1/N on the
        inverse, matching the standard convolution theorem).
    """

    def __init__(self, psf: np.ndarray, norm: str = "backward"):
        assert psf.ndim == 3, "PSF must be 3-D: (H, W, C)"
        self._psf = psf
        self._psf_shape = np.array(psf.shape[:2])   # (H, W)
        self._n_channels = psf.shape[2]
        self.norm = norm

        # Pad to at least 2H-1 × 2W-1 for linear (non-circular) convolution,
        # rounded to a "fast" FFT size.
        conv_shape = 2 * self._psf_shape - 1
        self._padded_shape = np.array([next_fast_len(int(s)) for s in conv_shape])

        # Center-pad window: PSF sits at [start:end] in the padded array.
        self._start = (self._padded_shape - self._psf_shape) // 2
        self._end   = self._start + self._psf_shape

        # Precompute FFT of the center-padded PSF and its conjugate.
        self._set_psf(psf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_psf(self, psf: np.ndarray):
        """Precompute H = rfft2(center_pad(psf)) and Hadj = conj(H)."""
        psf_pad   = self._pad(psf)
        self._H   = rfft2(psf_pad, axes=(0, 1), norm=self.norm)
        self._Hadj = np.conj(self._H)

    def _pad(self, v: np.ndarray) -> np.ndarray:
        """Center-pad (H, W, C) into a (pH, pW, C) array.

        The signal v is placed at [start:end] in the padded array; the rest
        is zero.  This centering is required so that ifftshift in convolve /
        deconvolve restores the correct spatial alignment.
        """
        ph, pw = self._padded_shape
        padded = np.zeros((ph, pw, v.shape[2]), dtype=v.dtype)
        padded[self._start[0]: self._end[0],
               self._start[1]: self._end[1], :] = v
        return padded

    def _crop(self, x: np.ndarray) -> np.ndarray:
        """Extract (H, W, C) from the center window [start:end] of x."""
        return x[self._start[0]: self._end[0],
                  self._start[1]: self._end[1], :]

    # ------------------------------------------------------------------
    # Public API – both methods expect PADDED inputs when called from ADMM
    # ------------------------------------------------------------------

    def convolve(self, x_padded: np.ndarray) -> np.ndarray:
        """Forward model: M x = ifftshift( irfft2( H · rfft2(x_padded) ) ).

        When called from outside ADMM (on an unpadded image), first call
        _pad(v) to get x_padded.  The final crop to (H, W) is NOT performed
        here; the ADMM optimises over the full padded image and crops only
        at the end via _form_image.

        Parameters
        ----------
        x_padded : ndarray, shape (pH, pW, C)
            Center-padded scene estimate.

        Returns
        -------
        out : ndarray, shape (pH, pW, C)
            Convolution result (still padded; crop if needed).
        """
        Fx  = rfft2(x_padded, axes=(0, 1), norm=self.norm)
        out = irfft2(self._H * Fx, axes=(0, 1),
                     s=tuple(self._padded_shape), norm=self.norm)
        return ifftshift(out, axes=(0, 1))

    def deconvolve(self, y_padded: np.ndarray) -> np.ndarray:
        """Adjoint: M^H y = ifftshift( irfft2( H* · rfft2(y_padded) ) ).

        Parameters
        ----------
        y_padded : ndarray, shape (pH, pW, C)

        Returns
        -------
        out : ndarray, shape (pH, pW, C)
        """
        Fy  = rfft2(y_padded, axes=(0, 1), norm=self.norm)
        out = irfft2(self._Hadj * Fy, axes=(0, 1),
                     s=tuple(self._padded_shape), norm=self.norm)
        return ifftshift(out, axes=(0, 1))

    def forward(self, v: np.ndarray) -> np.ndarray:
        """Convenience: apply forward model to an unpadded (H, W, C) scene.

        Pads v, convolves, and crops back to (H, W, C).  Useful for testing
        and reprojection error computation outside the ADMM loop.
        """
        return self._crop(self.convolve(self._pad(v)))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def H(self):
        return self._H

    @property
    def Hadj(self):
        return self._Hadj

    @property
    def padded_shape(self):
        return tuple(self._padded_shape)

    @property
    def psf_shape(self):
        return tuple(self._psf_shape)
