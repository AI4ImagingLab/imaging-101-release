# FPM Forward Model — reference excerpt

Coordinate/axis conventions and full method bodies. `M=N=1024`, `MM=NN=M*MAGimg=2048`, `MAGimg=2`.

```python
import torch
import torch.nn.functional as F

class FPMForwardModel:
    def __init__(self, Pupil0, kzz, ledpos_true, M, N, MAGimg):
        # Pupil0: (1,1,M,N) float; kzz: (1,M,N) complex; ledpos_true: (n_led,2) int
        self.Pupil0 = Pupil0; self.kzz = kzz; self.ledpos_true = ledpos_true
        self.M = M; self.N = N; self.MAGimg = MAGimg

    def compute_spectrum_mask(self, dz, led_num):
        dfmask = torch.exp(1j * self.kzz.repeat(dz.shape[0],1,1)
                           * dz[:,None,None].repeat(1, self.kzz.shape[1], self.kzz.shape[2]))
        dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
        a = self.Pupil0.repeat(len(dz), len(led_num),1,1) * torch.abs(dfmask)
        p = self.Pupil0.repeat(len(dz), len(led_num),1,1) * (torch.angle(dfmask) + 0)
        return a * torch.exp(1j * p)

    def get_led_coords(self, led_num):
        uo = self.ledpos_true[led_num, 0]; vo = self.ledpos_true[led_num, 1]
        x_0 = vo - self.M//2; x_1 = vo + self.M//2
        y_0 = uo - self.N//2; y_1 = uo + self.N//2
        return x_0, y_0, x_1, y_1

    def get_sub_spectrum(self, img_complex, led_num, spectrum_mask):
        x_0, y_0, x_1, y_1 = self.get_led_coords(led_num)
        O = torch.fft.fftshift(torch.fft.fft2(img_complex))
        tpx = (spectrum_mask.shape[-2]*self.MAGimg - O.shape[-2])//2
        tpy = (spectrum_mask.shape[-1]*self.MAGimg - O.shape[-1])//2
        O = F.pad(O, (tpx,tpx, tpy,tpy, 0,0), 'constant', 0)
        O_sub = torch.stack([O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))], dim=1)
        O_sub = O_sub * spectrum_mask
        o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
        return torch.abs(o_sub)

    def get_measured_amplitudes(self, Isum, led_num, n_z):
        oI = torch.sqrt(Isum[:, :, led_num])
        return oI.permute(2,0,1).unsqueeze(0).repeat(n_z,1,1,1)
```

Key subtleties to preserve:
- FFT chain is `fftshift(fft2(.))` forward and `ifft2(ifftshift(.))` inverse — do not drop the shifts.
- Sub-spectrum window is exactly M×N cropped from the centered padded spectrum at each LED's `(vo,uo)` center.
- The mask multiplies the cropped spectrum before the inverse transform; amplitude is `abs` of the inverse transform.
