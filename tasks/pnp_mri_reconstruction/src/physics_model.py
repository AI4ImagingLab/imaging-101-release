import numpy as np
import math
import decimal


class MRIForwardModel:
    """MRI forward model using subsampled 2D Fourier measurements.

    The forward model computes masked, centered FFT measurements:
        y = mask * fftshift(fft2(x)) / sqrt(N_pixels)

    The adjoint computes the inverse operation:
        x_adj = ifft2(ifftshift(mask * z)) * sqrt(N_pixels)
    """

    def __init__(self, mask):
        """Initialize with a sampling mask.

        Args:
            mask: ndarray (N, M) — boolean sampling mask in k-space
        """
        self.mask = mask
        self.sig_size = mask.shape

    def forward(self, x):
        """Forward model: image -> subsampled k-space.

        Args:
            x: ndarray (N, M) — image

        Returns:
            ndarray (N, M) — complex subsampled k-space
        """
        num_pix = self.mask.shape[0] * self.mask.shape[1]
        return self.mask * np.fft.fftshift(np.fft.fft2(x)) / math.sqrt(num_pix)

    def adjoint(self, z):
        """Adjoint operator: subsampled k-space -> image.

        Args:
            z: ndarray (N, M) — complex k-space data

        Returns:
            ndarray (N, M) — complex image
        """
        num_pix = self.mask.shape[0] * self.mask.shape[1]
        return np.fft.ifft2(np.fft.ifftshift(self.mask * z)) * math.sqrt(num_pix)

    def grad(self, x, y):
        """Gradient of the data fidelity term 0.5 * ||Ax - y||^2.

        Args:
            x: ndarray (N, M) — current image estimate
            y: ndarray (N, M) — observed k-space measurements

        Returns:
            tuple: (gradient, cost)
                gradient: ndarray (N, M) — real-valued gradient
                cost: float — data fidelity value
        """
        z = self.forward(x)
        residual = z - y
        g = self.adjoint(residual).real
        d = 0.5 * np.linalg.norm(residual.flatten("F")) ** 2
        return g, d

    def ifft_recon(self, y):
        """Naive IFFT reconstruction from subsampled k-space.

        Undoes the forward model normalization, zero-fills unsampled
        k-space locations, and applies inverse FFT.

        Args:
            y: ndarray (N, M) — subsampled k-space measurements (from forward())

        Returns:
            ndarray (N, M) — real-valued reconstruction
        """
        num_pix = self.mask.shape[0] * self.mask.shape[1]
        # Undo the 1/sqrt(N) normalization applied in forward()
        y_unnorm = y * math.sqrt(num_pix)
        return np.abs(np.fft.ifft2(np.fft.ifftshift(y_unnorm)))

    @staticmethod
    def generate_mask(image_size, num_lines):
        """Generate a radial sampling mask in k-space.

        Creates num_lines radial lines through the k-space center,
        evenly spaced in angle from 0 to pi.

        Args:
            image_size: ndarray or tuple (N, M) — image dimensions (must be even)
            num_lines: int — number of radial lines

        Returns:
            ndarray (N, M) — boolean mask
        """
        image_size = np.asarray(image_size)
        if image_size[0] % 2 != 0 or image_size[1] % 2 != 0:
            raise ValueError("Image dimensions must be even")

        center = np.ceil(image_size / 2) + 1
        freq_max = math.ceil(
            np.sqrt(np.sum(np.power(image_size / 2, 2), axis=0))
        )
        ang = np.linspace(0, math.pi, num=num_lines + 1)
        mask = np.zeros(image_size, dtype=bool)

        for ind_line in range(num_lines):
            ix = np.zeros(2 * freq_max + 1)
            iy = np.zeros(2 * freq_max + 1)
            ind = np.zeros(2 * freq_max + 1, dtype=bool)

            for i in range(2 * freq_max + 1):
                ix[i] = decimal.Decimal(
                    center[1] + (-freq_max + i) * math.cos(ang[ind_line])
                ).quantize(0, rounding=decimal.ROUND_HALF_UP)
            for i in range(2 * freq_max + 1):
                iy[i] = decimal.Decimal(
                    center[0] + (-freq_max + i) * math.sin(ang[ind_line])
                ).quantize(0, rounding=decimal.ROUND_HALF_UP)

            for k in range(2 * freq_max + 1):
                if (
                    ix[k] >= 1
                    and ix[k] <= image_size[1]
                    and iy[k] >= 1
                    and iy[k] <= image_size[0]
                ):
                    ind[k] = True
                else:
                    ind[k] = False

            ix = ix[ind].astype(np.int64)
            iy = iy[ind].astype(np.int64)

            for i in range(len(ix)):
                mask[iy[i] - 1][ix[i] - 1] = True

        return mask
