"""SIM parameter estimation: pattern frequencies, modulation depth, phase.

Before super-resolution reconstruction can be performed, three sets of
parameters must be recovered from the raw illumination-modulated frames:

1. Pattern wavevectors ``(k_x, k_y)`` for each illumination angle, found
   via cross-correlation of the phase-separated sidebands against the
   central (0th-order) band in Fourier space.
2. Modulation depth ``m`` (how strongly each sideband is modulated).
3. Phase offset between expected and observed sidebands.

The pipeline mirrors the MATLAB supplementary code for the Hessian-SIM
paper (Huang et al., 2018).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from .physics_model import pad_to_size, dft_conv, shift_otf


def _emd_decompose(signal):
    """1D empirical mode decomposition (IMFs as rows)."""
    from PyEMD import EMD as PyEMD_EMD
    emd_obj = PyEMD_EMD()
    emd_obj.FIXE_H = 0
    t = np.arange(len(signal))
    try:
        imfs = emd_obj.emd(signal.astype(np.float64), t)
    except Exception:
        return signal.reshape(1, -1)
    return imfs


def _phase_matrix(nphases, spjg, regul):
    total_spjg = sum(spjg)
    pm = np.zeros((nphases, nphases), dtype=np.complex128)
    for j in range(nphases):
        cum = sum(spjg[:j]) / total_spjg if j > 0 else 0.0
        pm[j, 0] = 1.0
        pm[j, 1] = np.exp(1j * regul * cum)
        pm[j, 2] = np.exp(-1j * regul * cum)
    return pm


def _compute_merit(H, H1, sp_center, sp_shifted, kx, ky, n):
    replcHtest = shift_otf(H1, kx, ky, n)
    mask = np.zeros_like(replcHtest, dtype=np.float64)
    mask[np.abs(replcHtest) > 0.9] = 1.0
    replcHtest = mask

    replch = shift_otf(H, kx, ky, n) * replcHtest

    sp_shifted_real = fftshift(ifft2(ifftshift(sp_shifted)))
    xx, yy = np.meshgrid(np.arange(2 * n), np.arange(2 * n))
    Ir = np.exp(1j * (kx * xx + ky * yy))
    replctest = fftshift(fft2(ifftshift(sp_shifted_real * Ir)))

    youhua = replctest * replcHtest * H
    he_num = np.abs(np.sum(np.conj(youhua) * (sp_center * replch)))
    he_den = np.sum(H1 * replcHtest)
    return he_num / (he_den + 1e-12)


def estimate_sim_parameters(raw_frames, otf, nangles, nphases, n,
                            pg, fanwei, regul, spjg, beishu_an=1):
    """Estimate SIM pattern frequencies.

    Returns pattern frequency coordinates ``(zuobiaox, zuobiaoy)`` in
    2n-grid pixel units, one per raw frame in the angle/phase block.
    """
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]

    fd = np.zeros((sy, sx, nframes), dtype=np.float64)
    for i in range(nframes):
        indices = list(range(i, nframes * beishu_an, nframes))
        fd[:, :, i] = raw_frames[indices].mean(axis=0)

    inv_pm = np.linalg.inv(_phase_matrix(nphases, spjg, regul))

    H = otf.copy().astype(np.float64)
    fc = int(np.ceil(220 * (n / 512)))
    ky_grid, kx_grid = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    H[kr > fc] = 0
    H = np.abs(H)
    H1 = (H != 0).astype(np.float64)
    H2 = H1.copy()

    fd_n = np.zeros((n, n, nframes), dtype=np.float64)
    py = (n - sy) // 2
    px = (n - sx) // 2
    for i in range(nframes):
        fd_n[py:py + sy, px:px + sx, i] = fd[:, :, i]
    DIbars = np.zeros((n, n, nframes), dtype=np.complex128)
    for i in range(nframes):
        DIbars[:, :, i] = fftshift(fft2(ifftshift(fd_n[:, :, i])))

    H9 = np.tile(H1[:, :, np.newaxis], (1, 1, nframes))
    DIbars = H9 * DIbars

    sp = np.zeros((n, n, nframes), dtype=np.complex128)
    for itheta in range(nangles):
        for j in range(nphases):
            for k in range(nphases):
                idx_in = itheta * nphases + k
                idx_out = itheta * nphases + j
                sp[:, :, idx_out] += inv_pm[j, k] * DIbars[:, :, idx_in]

    sp = sp / (np.abs(sp) + 1e-12)

    H_2n = pad_to_size(H, (2 * n, 2 * n))
    H1_2n = pad_to_size(H1, (2 * n, 2 * n))

    zuobiaox = np.full((nframes, 1), float(n))
    zuobiaoy = np.full((nframes, 1), float(n))

    for spi in range(0, nangles * (nphases - 1), 2):
        spi_m = spi + 1
        idx_center = int(np.ceil(spi_m / 2) * nphases - 1) - 1
        idx_shifted = int(np.ceil(spi_m / 2) + 2 * np.floor(spi_m / 2)) - 1

        sp_center = sp[:, :, idx_center]
        sp_shifted = sp[:, :, idx_shifted]

        H2_flip = H2[::-1, ::-1]
        cishu = dft_conv(H2, H2_flip)
        ci = cishu.copy()
        ci[np.abs(ci) < 0.9] = 0
        ci[ci != 0] = 1

        sp_conj_flip = np.conj(sp_shifted)[::-1, ::-1]
        jieguo = dft_conv(sp_center, sp_conj_flip)
        jieguo = jieguo * ci
        lihe = np.abs(jieguo / (cishu + 1e-12))

        lh, lw = lihe.shape
        ky_bp = np.arange(-lh // 2, lh // 2 + 1)[:lh]
        kx_bp = np.arange(-lw // 2, lw // 2 + 1)[:lw]
        KX, KY = np.meshgrid(kx_bp, ky_bp)
        kr_bp = np.sqrt(KX ** 2 + KY ** 2)
        mask_bp = (kr_bp >= (pg - fanwei)) & (kr_bp <= (pg + fanwei))
        lihe[~mask_bp] = 0

        peak = np.unravel_index(np.argmax(np.abs(lihe)), lihe.shape)
        maxx = float(peak[0])
        maxy = float(peak[1])

        sp_center_2n = pad_to_size(sp_center, (2 * n, 2 * n))
        sp_shifted_2n = pad_to_size(sp_shifted, (2 * n, 2 * n))

        kx0 = 2 * np.pi * (maxy - n) / (2 * n)
        ky0 = 2 * np.pi * (maxx - n) / (2 * n)
        he = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx0, ky0, n)

        eps_sub = 1e-5
        step_x = 1.0
        step_y = 1.0

        def _grad_dir_maxx():
            ky_m = 2 * np.pi * (maxx - eps_sub - n) / (2 * n)
            ky_p = 2 * np.pi * (maxx + eps_sub - n) / (2 * n)
            kx_c = 2 * np.pi * (maxy - n) / (2 * n)
            hm = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_c, ky_m, n)
            hp = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_c, ky_p, n)
            return -1 if hm > hp else (1 if hp > hm else None)

        def _grad_dir_maxy():
            kx_m = 2 * np.pi * (maxy - eps_sub - n) / (2 * n)
            kx_p = 2 * np.pi * (maxy + eps_sub - n) / (2 * n)
            ky_c = 2 * np.pi * (maxx - n) / (2 * n)
            hm = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_m, ky_c, n)
            hp = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_p, ky_c, n)
            return -1 if hm > hp else (1 if hp > hm else None)

        d = _grad_dir_maxx()
        flag_maxx = d if d is not None else 1
        d = _grad_dir_maxy()
        flag_maxy = d if d is not None else 1

        while step_x > 1e-4 or step_y > 1e-4:
            if step_x > 1e-4:
                d = _grad_dir_maxx()
                if d is not None:
                    flag_maxx = d
                else:
                    flag_maxx = -flag_maxx
                while step_x > 1e-4:
                    maxx_try = maxx + flag_maxx * step_x
                    ky_try = 2 * np.pi * (maxx_try - n) / (2 * n)
                    kx_try = 2 * np.pi * (maxy - n) / (2 * n)
                    he_try = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n,
                                            kx_try, ky_try, n)
                    if he_try > he:
                        he = he_try
                        maxx = maxx_try
                        break
                    else:
                        step_x *= 0.5

            if step_y > 1e-4:
                d = _grad_dir_maxy()
                if d is not None:
                    flag_maxy = d
                else:
                    flag_maxy = -flag_maxy
                while step_y > 1e-4:
                    maxy_try = maxy + flag_maxy * step_y
                    ky_try = 2 * np.pi * (maxx - n) / (2 * n)
                    kx_try = 2 * np.pi * (maxy_try - n) / (2 * n)
                    he_try = _compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n,
                                            kx_try, ky_try, n)
                    if he_try > he:
                        he = he_try
                        maxy = maxy_try
                        break
                    else:
                        step_y *= 0.5

        idx1 = spi_m + 1 + int(np.floor((spi_m - 1) / 2)) - 1
        idx2 = idx1 + 1
        zuobiaox[idx1, 0] = maxx
        zuobiaoy[idx1, 0] = maxy
        zuobiaox[idx2, 0] = 2 * n - maxx
        zuobiaoy[idx2, 0] = 2 * n - maxy

    return zuobiaox, zuobiaoy


def estimate_modulation_and_phase(raw_frames, otf, zuobiaox, zuobiaoy,
                                  nangles, nphases, n, regul, spjg,
                                  wavelength, beishu_an=1, bg=None):
    """Estimate modulation depth ``c6`` and phase offset ``angle6``."""
    n_512 = n
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]

    fc_ang = int(np.ceil(120 * (n_512 / 512)))
    fc_con = int(np.ceil(80 * (n_512 / 512))) if wavelength == 647 else int(np.ceil(105 * (n_512 / 512)))

    scale = n_512 / np.floor((zuobiaox[0, 0] + zuobiaoy[0, 0]) / 2)
    zuobiaox_512 = zuobiaox * scale
    zuobiaoy_512 = zuobiaoy * scale

    ky_grid, kx_grid = np.mgrid[-n_512 // 2:n_512 // 2, -n_512 // 2:n_512 // 2]
    k_r = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    H_ang = otf.copy().astype(np.float64)
    H_con = otf.copy().astype(np.float64)
    H_ang[k_r > fc_ang] = 0
    H_con[k_r > fc_con] = 0
    H_ang = np.abs(H_ang)
    H_con = np.abs(H_con)
    H1_ang = (H_ang != 0).astype(np.float64)
    H1_con = (H_con != 0).astype(np.float64)

    inv_pm = np.linalg.inv(_phase_matrix(nphases, spjg, regul))

    fd = np.zeros((sy, sx, nframes), dtype=np.float64)
    for i in range(nframes):
        indices = list(range(i, nframes * beishu_an, nframes))
        fd[:, :, i] = raw_frames[indices].mean(axis=0)

    if bg is not None:
        bgy, bgx = bg.shape
        cy, cx = bgy // 2, bgx // 2
        bg_crop = bg[cy - sy // 2:cy + sy // 2, cx - sx // 2:cx + sx // 2]
        if bg_crop.shape != (sy, sx):
            bg_crop = np.zeros((sy, sx))
    else:
        bg_crop = np.zeros((sy, sx))

    py = (n_512 - sy) // 2
    px = (n_512 - sx) // 2
    DIbars = np.zeros((n_512, n_512, nframes), dtype=np.complex128)
    for i in range(nframes):
        frame = fd[:, :, i] - bg_crop
        tmp = np.zeros((n_512, n_512), dtype=np.float64)
        tmp[py:py + sy, px:px + sx] = frame
        DIbars[:, :, i] = fftshift(fft2(ifftshift(tmp)))

    sp = np.zeros((n_512, n_512, nframes), dtype=np.complex128)
    for itheta in range(nangles):
        for j in range(nphases):
            for k in range(nphases):
                idx_in = itheta * nphases + k
                idx_out = itheta * nphases + j
                sp[:, :, idx_out] += inv_pm[j, k] * DIbars[:, :, idx_in]

    H_ang_2n = pad_to_size(H_ang, (2 * n_512, 2 * n_512))
    H_con_2n = pad_to_size(H_con, (2 * n_512, 2 * n_512))
    H1_ang_2n = pad_to_size(H1_ang, (2 * n_512, 2 * n_512))
    H1_con_2n = pad_to_size(H1_con, (2 * n_512, 2 * n_512))

    n_orders = nangles * (nphases - 1)
    c6 = np.zeros(n_orders)
    angle6 = np.zeros(n_orders)

    xx, yy = np.meshgrid(np.arange(2 * n_512), np.arange(2 * n_512))

    for spi in range(n_orders):
        spi_m = spi + 1
        idx_center = int(np.ceil(spi_m / 2)) * 3 - 1 - 1
        idx_shifted = int(np.ceil(spi_m / 2)) + 2 * int(np.floor(spi_m / 2)) - 1

        spzhongxin = pad_to_size(sp[:, :, idx_center], (2 * n_512, 2 * n_512))
        spzhongyiwei = pad_to_size(sp[:, :, idx_shifted], (2 * n_512, 2 * n_512))

        coord_idx = spi_m + 1 + int(np.floor((spi_m - 1) / 2)) - 1
        kytest = 2 * np.pi * (zuobiaox_512[coord_idx, 0] - n_512) / (2 * n_512)
        kxtest = 2 * np.pi * (zuobiaoy_512[coord_idx, 0] - n_512) / (2 * n_512)

        Ir = np.exp(1j * (kxtest * xx + kytest * yy))

        replcHtest_ang = shift_otf(H1_ang_2n, kxtest, kytest, n_512)
        mask_ang = np.zeros_like(replcHtest_ang, dtype=np.float64)
        mask_ang[np.abs(replcHtest_ang) > 0.9] = 1.0
        replcHtest_ang = mask_ang

        replcHtest_con = shift_otf(H1_con_2n, kxtest, kytest, n_512)
        mask_con = np.zeros_like(replcHtest_con, dtype=np.float64)
        mask_con[np.abs(replcHtest_con) > 0.9] = 1.0
        replcHtest_con = mask_con

        replch_ang = shift_otf(H_ang_2n, kxtest, kytest, n_512) * replcHtest_ang
        replch_con = shift_otf(H_con_2n, kxtest, kytest, n_512) * replcHtest_con

        sp_real = fftshift(ifft2(ifftshift(spzhongyiwei)))
        replctest = fftshift(fft2(ifftshift(sp_real * Ir)))

        youhua_ang = replctest * replcHtest_ang * H_ang_2n
        youhua_con = replctest * replcHtest_con * H_con_2n
        chongdie_ang = spzhongxin * replch_ang
        chongdie_con = spzhongxin * replch_con

        cm_ang = youhua_ang / (chongdie_ang + 1e-12)
        cm_con = youhua_con / (chongdie_con + 1e-12)

        angcm = np.angle(cm_ang)
        abscm = np.abs(cm_con)

        replc6_ang = replch_ang * H_ang_2n
        replc6_con = replch_con * H_con_2n

        a_ang = replc6_ang.ravel()
        mask_nz_ang = (a_ang != 0)
        phase_vals = angcm.ravel()[mask_nz_ang]

        a_con = replc6_con.ravel()
        mask_nz_con = (a_con != 0)
        mod_vals = abscm.ravel()[mask_nz_con]

        jdjd = 0.02
        jdc = 0.02
        f_bins = np.arange(-np.pi, np.pi + jdjd, jdjd)
        ff_bins = np.arange(0, 0.6 + jdc, jdc)

        g_hist = np.histogram(np.real(phase_vals), bins=f_bins)[0].astype(np.float64)
        gg_hist = np.histogram(np.real(mod_vals), bins=ff_bins)[0].astype(np.float64)

        if len(g_hist) > 10:
            try:
                imf = _emd_decompose(g_hist)
                if imf.shape[0] >= 5 and np.max(g_hist) < 50:
                    g_smooth = np.sum(imf[4:], axis=0)
                elif imf.shape[0] >= 4:
                    g_smooth = np.sum(imf[3:], axis=0)
                else:
                    g_smooth = g_hist
            except Exception:
                g_smooth = g_hist
        else:
            g_smooth = g_hist

        if len(gg_hist) > 10:
            try:
                imf2 = _emd_decompose(gg_hist)
                gg_smooth = np.sum(imf2, axis=0)
            except Exception:
                gg_smooth = gg_hist
        else:
            gg_smooth = gg_hist

        h_idx = np.argmax(g_smooth)
        angle6[spi] = -np.pi + (h_idx + 1) * jdjd

        hh_idx = np.where(gg_smooth == np.max(gg_smooth))[0]
        c6[spi] = jdc * np.mean(hh_idx + 1)

    return c6, angle6
