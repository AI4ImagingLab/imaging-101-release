"""
FPM Forward Model
=================

Fourier Ptychographic Microscopy forward model that simulates image formation:

1. The complex field at a given depth is transformed to Fourier space (FFT).
2. For each LED illumination, the corresponding sub-spectrum is extracted at the
   LED's spatial frequency offset.
3. A pupil mask and depth-dependent defocus propagation kernel are applied.
4. The sub-spectrum is inverse-transformed (iFFT) to yield the predicted
   low-resolution measurement amplitude.

Based on the original implementation by Haowen Zhou and Brandon Y. Feng.
"""

import numpy as np
import torch
import torch.nn.functional as F


class FPMForwardModel:
    """
    FPM forward model: complex field -> sub-spectrum measurements.

    Parameters
    ----------
    Pupil0 : torch.Tensor (1, 1, M, N)
        Pupil support mask.
    kzz : torch.Tensor (1, M, N)
        Angular spectrum propagation kernel (complex).
    ledpos_true : ndarray (n_leds, 2)
        LED positions in frequency space [x_pos, y_pos].
    M, N : int
        Raw measurement dimensions.
    MAGimg : int
        Upsampling ratio.
    """

    def __init__(self, Pupil0, kzz, ledpos_true, M, N, MAGimg):
        self.Pupil0 = Pupil0
        self.kzz = kzz
        self.ledpos_true = ledpos_true
        self.M = M
        self.N = N
        self.MAGimg = MAGimg

    def compute_spectrum_mask(self, dz, led_num):
        """
        Compute pupil * defocus mask for given z-depths and LEDs.

        Parameters
        ----------
        dz : torch.Tensor (n_z,)
            Depth positions.
        led_num : list of int
            LED indices.

        Returns
        -------
        spectrum_mask : torch.Tensor (n_z, n_leds, M, N) complex
        """
        dfmask = torch.exp(
            1j
            * self.kzz.repeat(dz.shape[0], 1, 1)
            * dz[:, None, None].repeat(1, self.kzz.shape[1], self.kzz.shape[2])
        )
        dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
        spectrum_mask_ampli = self.Pupil0.repeat(
            len(dz), len(led_num), 1, 1
        ) * torch.abs(dfmask)
        spectrum_mask_phase = self.Pupil0.repeat(len(dz), len(led_num), 1, 1) * (
            torch.angle(dfmask) + 0
        )  # 0 represents Pupil0 phase
        spectrum_mask = spectrum_mask_ampli * torch.exp(1j * spectrum_mask_phase)
        return spectrum_mask

    def get_led_coords(self, led_num):
        """
        Get sub-spectrum extraction coordinates for given LEDs.

        Parameters
        ----------
        led_num : list of int
            LED indices.

        Returns
        -------
        x_0, y_0, x_1, y_1 : arrays of int
            Start/end coordinates for sub-spectrum extraction.
        """
        uo = self.ledpos_true[led_num, 0]
        vo = self.ledpos_true[led_num, 1]
        x_0 = vo - self.M // 2
        x_1 = vo + self.M // 2
        y_0 = uo - self.N // 2
        y_1 = uo + self.N // 2
        return x_0, y_0, x_1, y_1

    def get_sub_spectrum(self, img_complex, led_num, spectrum_mask):
        """
        Extract sub-spectrum amplitudes for given LEDs.

        Parameters
        ----------
        img_complex : torch.Tensor (n_z, H, W) complex
            Predicted complex field.
        led_num : list of int
            LED indices.
        spectrum_mask : torch.Tensor (n_z, n_leds, M, N) complex
            Pupil * defocus mask.

        Returns
        -------
        oI_sub : torch.Tensor (n_z, n_leds, M, N) float
            Predicted amplitudes.
        """
        x_0, y_0, x_1, y_1 = self.get_led_coords(led_num)

        O = torch.fft.fftshift(torch.fft.fft2(img_complex))
        to_pad_x = (spectrum_mask.shape[-2] * self.MAGimg - O.shape[-2]) // 2
        to_pad_y = (spectrum_mask.shape[-1] * self.MAGimg - O.shape[-1]) // 2
        O = F.pad(O, (to_pad_x, to_pad_x, to_pad_y, to_pad_y, 0, 0), "constant", 0)

        O_sub = torch.stack(
            [O[:, x_0[i] : x_1[i], y_0[i] : y_1[i]] for i in range(len(led_num))],
            dim=1,
        )
        O_sub = O_sub * spectrum_mask
        o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
        oI_sub = torch.abs(o_sub)

        return oI_sub

    def get_measured_amplitudes(self, Isum, led_num, n_z):
        """
        Get measured amplitudes for given LEDs.

        Parameters
        ----------
        Isum : torch.Tensor (M, N, n_leds)
            Normalized measurements.
        led_num : list of int
            LED indices.
        n_z : int
            Number of z-planes.

        Returns
        -------
        oI_cap : torch.Tensor (n_z, n_leds, M, N) float
        """
        oI_cap = torch.sqrt(Isum[:, :, led_num])
        oI_cap = oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(n_z, 1, 1, 1)
        return oI_cap
