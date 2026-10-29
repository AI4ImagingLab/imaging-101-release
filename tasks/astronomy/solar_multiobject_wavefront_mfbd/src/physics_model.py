"""Local pupil, KL wavefront and explicit Fourier variable-projection mathematics.

Adapted from torchmfbd by A. Asensio Ramos and contributors; original legal text
and source revision are preserved in test/validation.md.
"""
import numpy as np
import torch
from scipy import ndimage, special

DIAMETER_CM = 100.0
WAVELENGTH_ANGSTROM = 8542.0
PIXEL_ARCSEC = 0.059
MODE_COUNT = 54
PRIOR_POWER = 100.0


def noll_indices(index):
    """Convert one-based Noll index to radial order and signed azimuthal order."""
    radial = int((-1 + np.sqrt(8*(index-1)+1))/2)
    remainder = index - radial*(radial+1)//2
    sign = 1 if index % 2 == 0 else -1
    azimuth = 2 * ((remainder + radial % 2)//2) - radial % 2
    return radial, sign * azimuth


def build_optics(mixing, size=88, device='cpu'):
    """Construct discrete pupil, explicit KL basis and diffraction cutoff mask."""
    wavelength = np.float32(WAVELENGTH_ANGSTROM)
    overfill = 206265.0 * wavelength * 1e-8 / (DIAMETER_CM * PIXEL_ARCSEC)
    pupil_coordinate = np.arange(-size/2, size/2, dtype='float64')
    xx, yy = np.meshgrid(pupil_coordinate, pupil_coordinate)
    pupil = (np.sqrt(xx**2 + yy**2)/(size/2) <= 1.0/overfill).astype('float32')
    coordinate = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(coordinate, coordinate)
    radius = overfill * np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)
    support = radius <= 1.0
    zernikes = {}
    for index in np.unique(mixing[:, 1]).astype(int):
        radial, signed = noll_indices(index)
        azimuth = abs(signed)
        order = (radial-azimuth)//2
        polynomial = (-1)**order * radius**azimuth * special.eval_jacobi(
            order, azimuth, 0, 1-2*radius**2)
        if signed == 0:
            value = np.sqrt(radial+1) * polynomial
        elif signed > 0:
            value = np.sqrt(2)*np.sqrt(radial+1)*polynomial*np.cos(azimuth*angle)
        else:
            value = np.sqrt(2)*np.sqrt(radial+1)*polynomial*np.sin(azimuth*angle)
        zernikes[index] = value
    basis = np.zeros((MODE_COUNT, size, size))
    # Preserve the selected upstream convention: KL indices 2..54, then a null slot.
    for slot in range(MODE_COUNT-1):
        rows = mixing[mixing[:, 0] == slot+2]
        for _, zernike, coefficient in rows:
            basis[slot] += coefficient * zernikes[int(zernike)] * support
    cutoff = DIAMETER_CM / (wavelength * 1e-8) / 206265.0
    frequency = np.fft.fftfreq(size, d=PIXEL_ARCSEC) / cutoff
    fx, fy = np.meshgrid(frequency, frequency)
    rho = torch.tensor(np.sqrt(fx**2 + fy**2).astype('float32'), device=device)
    mask = torch.zeros_like(rho)
    mask[rho <= 0.85] = 1.0
    transition = (rho > 0.85) & (rho <= 0.90)
    mask[transition] = torch.cos(np.pi/2 * (rho[transition]-0.85)/(0.90-0.85))
    return {'pupil': torch.tensor(pupil, device=device),
            'basis': torch.tensor(basis.astype('float32'), device=device),
            'frequency_radius': rho, 'diffraction_mask': mask}


def canonical_modes(modes):
    """Fix the image-translation gauge by setting first-frame tip/tilt to zero."""
    result = modes.clone()
    result[..., :2] = result[..., :2] - modes[:, :1, :2]
    return result


def pupil_to_psf(modes, pupil, basis, active=None):
    """Return unit-flux origin-centered PSFs and orthonormal FFT OTFs."""
    active = modes.shape[-1] if active is None else active
    wavefront = torch.einsum('ijk,klm->ijlm', modes[..., :active], basis[:active])
    phase = pupil[None, None] * torch.exp(1j * wavefront)
    field = torch.fft.fft2(phase, norm='ortho')
    power = (torch.conj(field) * field).real
    psf = power / torch.sum(power, dim=(-1, -2), keepdim=True)
    return psf, torch.fft.fft2(psf, norm='ortho')


def reduced_loss(frame_fts, otfs, sigmas, diffraction_mask):
    """Selected upstream joint-MAP reduced loss with constant object PSD 100."""
    loss = torch.tensor(0.0, device=frame_fts[0].device, dtype=frame_fts[0].real.dtype)
    for frames, otf, sigma in zip(frame_fts, otfs, sigmas):
        variance = torch.mean(sigma**2)
        denominator = variance + PRIOR_POWER * torch.sum(otf * torch.conj(otf), dim=1)
        data_power = torch.sum(frames * torch.conj(frames), dim=1)
        cross = torch.sum(frames * torch.conj(otf), dim=1)
        numerator = PRIOR_POWER * cross * torch.conj(cross)
        local = 0.5 * (data_power - numerator/denominator) / variance
        local *= diffraction_mask[None]
        # The bound reference excludes both zero-frequency axes, not only DC.
        loss += torch.mean(local[:, 1:, 1:].real)
    return loss


def scharmer_filter(normal, cross, variance, diffraction_mask):
    """Median-denoised spectral support and eight-connected central component."""
    ratio = (normal/(torch.conj(cross)*cross)).real
    shifted = torch.fft.fftshift(ratio, dim=(-2, -1)).detach().cpu().numpy()
    smoothed = ndimage.median_filter(shifted, size=(1, 3, 3), mode='wrap')
    variance_array = variance.detach().cpu().numpy()
    if variance_array.ndim:
        variance_array = variance_array[:, None, None]
    filt = 1.0 - smoothed * variance_array
    filt[filt < 0.2] = 0.0
    filt[filt > 1.0] = 1.0
    masks = np.zeros_like(filt, dtype='float32')
    for index, value in enumerate(filt):
        center = value.shape[0]//2, value.shape[1]//2
        image = 1.0-value
        admitted = np.abs(image-image[center]) <= 0.9
        labels, _ = ndimage.label(admitted, structure=np.ones((3, 3)))
        central = labels[center]
        component = (labels == central) if central else np.zeros_like(admitted)
        masks[index] = np.fft.fftshift(component)
    return torch.tensor(masks, device=normal.device) * diffraction_mask[None]


def project_objects(frame_fts, otfs, sigmas, diffraction_mask):
    """Noise-weighted analytic object projection followed by spectral support filter."""
    objects = []
    for frames, otf, sigma in zip(frame_fts, otfs, sigmas):
        variance = torch.mean(sigma**2, dim=1)
        weight = variance[:, None, None, None]/(sigma[:, :, None, None]**2 + 1e-10)
        normal = torch.sum(weight * torch.conj(otf) * otf, dim=1)
        cross = torch.sum(weight * torch.conj(otf) * frames, dim=1)
        mask = scharmer_filter(normal, cross, torch.mean(sigma**2), diffraction_mask)
        filtered = cross/(normal+1e-10)*mask
        objects.append(torch.fft.ifft2(filtered).real)
    return torch.stack(objects)
