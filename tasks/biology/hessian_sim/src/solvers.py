"""Inverse solvers for Hessian-SIM.

Three solvers are provided:

* ``wiener_sim_reconstruct`` — band-unmixing Wiener-filtered SIM
  reconstruction producing a 2x super-resolved image.
* ``hessian_denoise`` — split-Bregman solver with a second-order (Hessian)
  regulariser used to suppress noise while preserving fine continuous
  structures.
* ``tv_denoise`` — split-Bregman solver with an anisotropic first-order
  total-variation regulariser (baseline comparison).
* ``running_average`` — temporal smoothing of the reconstructed stack.

Split-Bregman minimises

    argmin_x   λ Σ_k ||Φ_k x||_1   +   (μ/2) ||x - y||_2²

with ``Φ_k`` a first-order (TV) or second-order (Hessian) difference operator.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift

from .physics_model import pad_to_size, shift_otf


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def forward_diff(x, step, dim):
    """Forward finite difference along ``dim`` with periodic BC."""
    return (np.roll(x, -1, axis=dim) - x) / step


def back_diff(x, step, dim):
    """Backward finite difference along ``dim`` with periodic BC."""
    return (x - np.roll(x, 1, axis=dim)) / step


def wiener_sim_reconstruct(raw_frames, otf, zuobiaox, zuobiaoy, c6, angle6,
                           nangles, nphases, n, weilac, regul, spjg,
                           beishu_re=1, starframe=0, bg=None):
    """Band-unmixing Wiener-filtered SIM reconstruction.

    Returns a ``(n_timepoints, 2*sy, 2*sx)`` stack of super-resolved frames.
    """
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]
    total_frames = raw_frames.shape[0]

    image_size = max(sy, sx)
    if image_size <= 256:
        otfx = 256
    elif image_size <= 512:
        otfx = 512
    else:
        otfx = image_size

    zuobiaox_r = zuobiaox * (otfx / n)
    zuobiaoy_r = zuobiaoy * (otfx / n)
    n_work = otfx

    num_images = total_frames - starframe
    n_timepoints = num_images // (nframes * beishu_re)

    xishu = np.ones(nframes)
    for itheta in range(nangles):
        c6_pair = c6[itheta * (nphases - 1):(itheta + 1) * (nphases - 1)]
        avg_inv = np.mean(1.0 / (c6_pair + 1e-12))
        xishu[itheta * nphases + 1] = avg_inv
        xishu[itheta * nphases + 2] = avg_inv

    deph = np.zeros(nangles)
    for itheta in range(nangles):
        a6_pair = angle6[itheta * (nphases - 1):(itheta + 1) * (nphases - 1)]
        deph[itheta] = np.sign(a6_pair[0]) * np.mean(np.abs(a6_pair))

    fc = int(np.ceil(200 * (n_work / 512)))
    if otf.shape[0] != n_work or otf.shape[1] != n_work:
        from scipy.ndimage import zoom as ndizoom
        otf_r = ndizoom(otf, n_work / otf.shape[0], order=1)
    else:
        otf_r = otf.copy()

    ky_grid, kx_grid = np.mgrid[-n_work // 2:n_work // 2, -n_work // 2:n_work // 2]
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    jiequ = np.ones((n_work, n_work))
    jiequ[kr > fc] = 0

    H = otf_r.astype(np.float64) * jiequ
    H = H / (H.max() + 1e-12)
    H1 = (H != 0).astype(np.float64)

    Hk = pad_to_size(H, (2 * n_work, 2 * n_work))
    H1_2n = pad_to_size(H1, (2 * n_work, 2 * n_work))

    xx, yy = np.meshgrid(np.arange(2 * n_work), np.arange(2 * n_work))
    Irtest = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
    replcHtest = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.float64)
    replch = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)

    for ii in range(nframes):
        kytest = 2 * np.pi * (zuobiaox_r[ii, 0] - n_work) / (2 * n_work)
        kxtest = 2 * np.pi * (zuobiaoy_r[ii, 0] - n_work) / (2 * n_work)
        Irtest[:, :, ii] = np.exp(1j * (kxtest * xx + kytest * yy))
        replcHtest[:, :, ii] = np.abs(shift_otf(H1_2n, kxtest, kytest, n_work))
        replch[:, :, ii] = shift_otf(Hk, kxtest, kytest, n_work)

    replcHtest[np.abs(replcHtest) > 0.9] = 1.0
    replcHtest[np.abs(replcHtest) != 1.0] = 0.0
    replch = replch * replcHtest

    re = replch.copy()
    absre = np.abs(re)
    absre_mask = (absre != 0).astype(np.float64)
    re = re * absre_mask
    re[re == 0] = 1e19
    re = re / np.abs(re)

    reh = np.abs(replch)
    hs = np.sum(reh ** 2, axis=2)

    plong = int(np.floor(
        np.sum(np.sqrt((zuobiaox_r[:, 0] - zuobiaox_r[0, 0]) ** 2 +
                       (zuobiaoy_r[:, 0] - zuobiaoy_r[0, 0]) ** 2)) / (2 * nangles)
    ))
    k_max = plong + fc
    ky_2n, kx_2n = np.mgrid[-n_work:n_work, -n_work:n_work]
    kr_2n = np.sqrt(kx_2n ** 2 + ky_2n ** 2)
    bhs = np.cos(np.pi * kr_2n / (2 * k_max))
    bhs[kr_2n > k_max] = 0

    total_spjg = sum(spjg)
    inv_phase_matrices = np.zeros((nphases, nphases, nangles), dtype=np.complex128)
    for itheta in range(nangles):
        phi_tmp = np.linspace(0 + deph[itheta], regul + deph[itheta], total_spjg + 1)
        phi = np.zeros(nphases)
        phi[0] = phi_tmp[0]
        phi[1] = phi_tmp[int(spjg[0])]
        phi[2] = phi_tmp[int(spjg[0] + spjg[1])]

        pm = np.zeros((nphases, nphases), dtype=np.complex128)
        for j in range(nphases):
            pm[j, 0] = 1.0
            pm[j, 1] = np.exp(1j * phi[j])
            pm[j, 2] = np.exp(-1j * phi[j])
        inv_phase_matrices[:, :, itheta] = np.linalg.inv(pm)

    sig = 0.25
    x_mask = np.arange(1, sx + 1)
    y_mask = np.arange(1, sy + 1)[:, np.newaxis]
    mask = (_sigmoid(sig * x_mask) - _sigmoid(sig * (x_mask - sx - 1))) * \
           (_sigmoid(sig * y_mask) - _sigmoid(sig * (y_mask - sy - 1)))
    mask9 = np.tile(mask[:, :, np.newaxis] ** 3, (1, 1, nframes))

    if bg is not None:
        bgy, bgx = bg.shape
        cy, cx = bgy // 2, bgx // 2
        bg_crop = bg[cy - sy // 2:cy + sy // 2, cx - sx // 2:cx + sx // 2]
        if bg_crop.shape != (sy, sx):
            bg_crop = np.zeros((sy, sx))
    else:
        bg_crop = np.zeros((sy, sx))

    result_stack = []
    zhen = starframe
    while zhen <= num_images - nframes * beishu_re + starframe:
        D = raw_frames[zhen:zhen + nframes * beishu_re].astype(np.float64)
        for i in range(D.shape[0]):
            D[i] -= bg_crop

        fd = np.zeros((sy, sx, nframes), dtype=np.float64)
        for i in range(nframes):
            indices = list(range(i, nframes * beishu_re, nframes))
            fd[:, :, i] = D[indices].mean(axis=0)

        fd *= mask9

        py = (n_work - sy) // 2
        px = (n_work - sx) // 2
        DIbar = np.zeros((n_work, n_work, nframes), dtype=np.complex128)
        for i in range(nframes):
            tmp = np.zeros((n_work, n_work), dtype=np.float64)
            tmp[py:py + sy, px:px + sx] = fd[:, :, i]
            DIbar[:, :, i] = fftshift(fft2(ifftshift(tmp)))

        sp = np.zeros((n_work, n_work, nframes), dtype=np.complex128)
        for itheta in range(nangles):
            inv_pm = inv_phase_matrices[:, :, itheta]
            for j in range(nphases):
                for k in range(nphases):
                    idx_in = itheta * nphases + k
                    idx_out = itheta * nphases + j
                    sp[:, :, idx_out] += inv_pm[j, k] * DIbar[:, :, idx_in]

        jiequ9 = np.tile(jiequ[:, :, np.newaxis], (1, 1, nframes))
        sp = sp * jiequ9

        spsim = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
        retirff = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
        for ii in range(nframes):
            sp_padded = pad_to_size(sp[:, :, ii], (2 * n_work, 2 * n_work))
            spsim[:, :, ii] = sp_padded
            sp_real = fftshift(ifft2(ifftshift(sp_padded)))
            retirff[:, :, ii] = fftshift(fft2(ifftshift(sp_real * Irtest[:, :, ii])))

        retirff = retirff * replcHtest
        retirff = retirff / re

        tmprc = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
        for t in range(nframes):
            tmprc[:, :, t] = (xishu[t] * retirff[:, :, t] * np.conj(reh[:, :, t])) / \
                             (hs + 0.005 * nangles * weilac ** 2)

        dr = np.sum(tmprc, axis=2)
        drr = dr * bhs

        fimage = fftshift(ifft2(ifftshift(drr)))

        cy, cx = n_work, n_work
        result = np.real(fimage[cy - sy:cy + sy, cx - sx:cx + sx])
        result[result < 0] = 0
        result_stack.append(result.astype(np.float32))

        zhen += nframes * beishu_re

    return np.array(result_stack)


def hessian_denoise(stack, mu=150.0, sigma_z=1.0, n_iter=100):
    """Split-Bregman Hessian (second-order) denoising of a 3D stack.

    Applies six second-order difference operators (d_xx, d_yy, d_zz, and
    the three cross-terms) in an L1 penalty solved by soft-thresholding +
    FFT-domain inverse.
    """
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    nz, sy_s, sx_s = stack.shape
    nz_orig = nz

    if nz < 3:
        sigma_z = 0.0
        pad_z = 3 - nz
        stack = np.concatenate([stack, np.tile(stack[-1:], (pad_z, 1, 1))], axis=0)
        nz = 3

    y = stack.astype(np.float32)
    ymax = y.max()
    if ymax > 0:
        y = y / ymax

    sizex = (nz, sy_s, sx_s)
    siranu = mu
    lamda = mu * 20
    zbei = sigma_z

    kernel = np.zeros(sizex); kernel[0, 0, :3] = [1, -2, 1]
    Frefft = fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex); kernel[0, :3, 0] = [1, -2, 1]
    Frefft = Frefft + fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex); kernel[:3, 0, 0] = [1, -2, 1]
    Frefft = Frefft + (zbei ** 2) * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex); kernel[0, :2, :2] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex); kernel[:2, 0, :2] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * zbei * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex); kernel[:2, :2, 0] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * zbei * fftn(kernel) * np.conj(fftn(kernel))

    divide = np.real((siranu / lamda) + Frefft).astype(np.float32)

    b = [np.zeros(sizex, dtype=np.float32) for _ in range(6)]
    x = np.zeros(sizex, dtype=np.float32)

    for it in range(n_iter):
        frac = (siranu / lamda) * y
        frac_f = fftn(frac)
        if it > 0:
            x = np.real(ifftn(frac_f / divide)).astype(np.float32)
        else:
            x = np.real(ifftn(frac_f / (siranu / lamda))).astype(np.float32)

        frac = (siranu / lamda) * y

        u = back_diff(forward_diff(x, 1, 2), 1, 2)
        s = np.abs(u + b[0]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[0])
        b[0] += u - d
        frac += back_diff(forward_diff(d - b[0], 1, 2), 1, 2)

        u = back_diff(forward_diff(x, 1, 1), 1, 1)
        s = np.abs(u + b[1]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[1])
        b[1] += u - d
        frac += back_diff(forward_diff(d - b[1], 1, 1), 1, 1)

        u = back_diff(forward_diff(x, 1, 0), 1, 0)
        s = np.abs(u + b[2]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[2])
        b[2] += u - d
        frac += (zbei ** 2) * back_diff(forward_diff(d - b[2], 1, 0), 1, 0)

        u = forward_diff(forward_diff(x, 1, 2), 1, 1)
        s = np.abs(u + b[3]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[3])
        b[3] += u - d
        frac += 2 * back_diff(back_diff(d - b[3], 1, 1), 1, 2)

        u = forward_diff(forward_diff(x, 1, 2), 1, 0)
        s = np.abs(u + b[4]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[4])
        b[4] += u - d
        frac += 2 * zbei * back_diff(back_diff(d - b[4], 1, 0), 1, 2)

        u = forward_diff(forward_diff(x, 1, 1), 1, 0)
        s = np.abs(u + b[5]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[5])
        b[5] += u - d
        frac += 2 * zbei * back_diff(back_diff(d - b[5], 1, 0), 1, 1)

    x[x < 0] = 0
    x = x[:nz_orig] * ymax
    return x


def tv_denoise(stack, mu=150.0, n_iter=100):
    """Split-Bregman anisotropic total-variation denoising (2D in-plane only)."""
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    nz, sy_s, sx_s = stack.shape
    y = stack.astype(np.float64)
    ymax = y.max()
    if ymax > 0:
        y = y / ymax

    sizex = (nz, sy_s, sx_s)
    lamda = mu * 20
    siranu = mu

    kernel_x = np.zeros(sizex); kernel_x[0, 0, :2] = [1, -1]
    xfft = fftn(kernel_x) * np.conj(fftn(kernel_x))

    kernel_y = np.zeros(sizex); kernel_y[0, :2, 0] = [1, -1]
    yfft = fftn(kernel_y) * np.conj(fftn(kernel_y))

    b7 = np.zeros(sizex)
    b8 = np.zeros(sizex)
    d7 = np.zeros(sizex)
    d8 = np.zeros(sizex)
    x = np.zeros(sizex)

    for it in range(n_iter):
        frac = (siranu / lamda) * y
        frac -= back_diff(d7 - b7, 1, 2)
        frac -= back_diff(d8 - b8, 1, 1)
        frac_f = fftn(frac)
        divide = (siranu / lamda) + xfft + yfft
        x = np.real(ifftn(frac_f / divide))

        u_x = forward_diff(x, 1, 2)
        u_y = forward_diff(x, 1, 1)

        s7 = np.abs(u_x + b7) - 1.0 / lamda
        s7[s7 < 0] = 0
        d7 = s7 * np.sign(u_x + b7)

        s8 = np.abs(u_y + b8) - 1.0 / lamda
        s8[s8 < 0] = 0
        d8 = s8 * np.sign(u_y + b8)

        b7 += u_x - d7
        b8 += u_y - d8
        x[x < 0] = 0

    x = x * ymax
    return x.astype(np.float32)


def running_average(stack, window=3):
    """Centred sliding-window temporal average; edge frames left as zeros."""
    if stack.ndim < 3 or stack.shape[0] < window:
        return stack

    nz = stack.shape[0]
    result = np.zeros_like(stack)
    half_before = (window - 1) // 2
    half_after = window // 2

    for i in range(half_before, nz - half_after):
        result[i] = np.mean(stack[i - half_before:i + half_after + 1], axis=0)

    return result
