# PSF convolution

```python
import torch, torch.nn.functional as F

kH, kW = psf.shape                    # (32, 32)
psf_t = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)
ph_lo = (kH - 1)//2; ph_hi = kH - 1 - ph_lo
pw_lo = (kW - 1)//2; pw_hi = kW - 1 - pw_lo

def psf_blur(z):
    z_pad = F.pad(z, (pw_lo, pw_hi, ph_lo, ph_hi), mode='reflect')
    return F.conv2d(z_pad, psf_t, padding=0)
```

The PSF is already unit-sum; do not renormalize.
