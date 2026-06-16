"""FWI inversion: objective function, gradient computation, NCG optimizer.

Adapted from CBS_FWI_torch/utils/GetGrid_utils.py, inversion.py,
Optimizer_ncg_utils.py, and Optimize_utils.py.
"""

import logging
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Dict, Optional, List

from src.physics_model import solve_all_sources


# ---------------------------------------------------------------------------
# Gradient smoothing
# ---------------------------------------------------------------------------

def create_gaussian_kernel(kernel_size: int = 9, sigma: float = 1.0) -> torch.Tensor:
    """Create 2D Gaussian kernel for gradient smoothing."""
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    x = torch.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = x / x.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


# ---------------------------------------------------------------------------
# Source estimation and gradient
# ---------------------------------------------------------------------------

def create_objective(all_u_size, freq, dobs_masked, mask_esi, mask_misfit,
                     R, ix, iy, sigma, dh=50, ppw=8, lamb=1,
                     boundary_widths=20, born_max=500, energy_threshold=1e-5):
    """Create the FWI objective function closure.

    Args:
        all_u_size: [nx, ny, n_rec]
        freq: frequency in 100kHz units
        dobs_masked: (n_rec, n_rec) masked observations
        mask_esi: (n_rec, n_rec) source estimation mask
        mask_misfit: (n_rec, n_rec) misfit mask
        R: sparse (n_rec, nx*ny) restriction operator
        ix, iy: (n_rec,) 1-indexed receiver positions
        sigma: Gaussian blur sigma for gradient
        dh, ppw, lamb, boundary_widths, born_max, energy_threshold: CBS params

    Returns:
        get_grad: callable(slowness, fscale, gscale) -> (J, G)
    """
    omg = 2 * torch.pi * freq
    n_rec = all_u_size[2]
    kernel_size = 9
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel.expand(1, 1, -1, -1)

    def get_grad(slowness, fscale=1, gscale=1):
        start_time = time.time()
        vp = 1 / slowness

        # Forward solve
        all_u = solve_all_sources(vp, ix, iy, freq, dh, ppw, lamb,
                                  boundary_widths, born_max, energy_threshold)

        # Restrict to receiver locations
        all_u_flatten = all_u.reshape(-1, n_rec)
        dsrc_real = torch.sparse.mm(R, all_u_flatten.real)
        dsrc_imag = torch.sparse.mm(R, all_u_flatten.imag)
        dsrc = dsrc_real + 1j * dsrc_imag
        dsrc = dsrc * mask_esi

        # Estimate source intensity
        den = dsrc * torch.conj(dsrc)
        num = dobs_masked * torch.conj(dsrc)
        sval_all = torch.tensor([
            (torch.sum(num[:, idx]) / torch.sum(den[:, idx]))
            for idx in range(n_rec)
        ]).cuda()

        dobs_sim = dsrc * sval_all[None, ...] * mask_misfit
        u_source_intense = all_u * sval_all[None, None, ...]

        # Misfit
        dobs_misfit = dobs_sim - dobs_masked
        J_column = torch.norm(dobs_misfit, p=2, dim=0, keepdim=True)
        J = torch.sum(torch.square(J_column))

        # Gradient via adjoint
        G = torch.zeros(all_u_size[:-1], dtype=torch.float64).cuda()
        q = [
            torch.sum(all_u * torch.conj(dobs_misfit[:, isrc]), dim=2)[..., None]
            for isrc in range(n_rec)
        ]
        q = torch.cat(q, dim=2)
        G = G - torch.real(torch.sum((u_source_intense * q), dim=2)) * (
            omg**2 / vp * 2
        )
        G = G.T / 1e10

        # Gaussian blur
        G = G.unsqueeze(0).unsqueeze(0).float()
        G_blurred = F.conv2d(G, gaussian_kernel.cuda(), padding=kernel_size // 2)
        G = G_blurred.squeeze(0).squeeze(0)
        G = G / gscale
        J = J / fscale / 1e9

        elapsed = time.time() - start_time
        logging.info(f"forward time {elapsed:.1f}s")
        return J, G

    return get_grad


# ---------------------------------------------------------------------------
# More-Thuente line search (from Optimizer_ncg_utils.py)
# ---------------------------------------------------------------------------

def _cstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """Cubic step computation for More-Thuente line search."""
    p66 = 0.66
    info = torch.tensor(0)
    if (
        brackt & ((stp <= min(stx, sty)) | (stp >= max(stx, sty)))
        | (dx * (stp - stx) >= 0.0)
        | (stpmax < stpmin)
    ):
        return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info

    sgnd = dp * (dx / torch.abs(dx))

    if fp > fx:
        info = torch.tensor(1)
        bound = torch.tensor(1)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2) * (stp - stx)
        if torch.abs(stpc - stx) < torch.abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2
        brackt = torch.tensor(1)
    elif sgnd < 0.0:
        info = torch.tensor(2)
        bound = torch.tensor(0)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if torch.abs(stpc - stp) > torch.abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = torch.tensor(1)
    elif torch.abs(dp) < torch.abs(dx):
        info = torch.tensor(3)
        bound = torch.tensor(1)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt(
            torch.max(torch.tensor(0.0), (theta / s) ** 2 - (dx / s) * (dp / s))
        )
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if (r < 0.0) & (gamma != 0.0):
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            if abs(stp - stpc) < abs(stp - stpq):
                stpf = stpc
            else:
                stpf = stpq
        else:
            if abs(stp - stpc) > abs(stp - stpq):
                stpf = stpc
            else:
                stpf = stpq
    else:
        info = torch.tensor(4)
        bound = torch.tensor(0)
        if brackt:
            theta = 3 * (fp - fy) / (sty - stp) + dy + dp
            s = torch.norm(torch.tensor([theta, dy, dp]), float("inf"))
            gamma = s * torch.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    stpf = min(stpmax, stpf)
    stpf = max(stpmin, stpf)
    stp = stpf
    if brackt & bound:
        if sty > stx:
            stp = min(stx + p66 * (sty - stx), stp)
        else:
            stp = max(stx + p66 * (sty - stx), stp)

    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info


def _cvsrch(fcn, x, f, g, stp, s, ub, lb, maxfev=5):
    """More-Thuente line search with bound constraints."""
    xtol = 1e-15
    ftol = 1e-4
    gtol = 1e-2
    stpmin = 1e-15
    stpmax = 1e15

    nfev = 0
    p5 = 0.5
    p66 = 0.66
    xtrapf = 4.0
    info = 0
    infoc = 1

    dginit = torch.dot(torch.conj(g.T.flatten()), torch.conj(s.flatten()))
    if dginit >= 0.0:
        return x, f, g, stp, info, nfev

    brackt = False
    stage1 = True
    finit = f
    dgtest = ftol * dginit
    width = stpmax - stpmin
    width1 = 2 * width
    wa = x.clone()

    stx = 0.0
    fx_ls = finit
    dgx = dginit
    sty = 0.0
    fy = finit
    dgy = dginit

    while True:
        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stx
            stmax = stp + xtrapf * (stp - stx)

        stp = max(stp, stpmin)
        stp = min(stp, stpmax)

        if (
            (brackt and (stp <= stmin or stp >= stmax))
            or nfev >= maxfev
            or infoc == 0
            or (brackt and stmax - stmin <= xtol * stmax)
        ):
            stp = stx

        x = wa + stp * s
        x = torch.minimum(x, ub)
        x = torch.maximum(x, lb)

        f, g = fcn(x)
        nfev += 1
        dg = torch.dot(torch.conj(g.T.flatten()), torch.conj(s.flatten()))
        ftest1 = finit + stp * dgtest

        if (brackt and (stp <= stmin or stp >= stmax)) or infoc == 0:
            info = 6
        if stp == stpmax and f <= ftest1 and dg <= dgtest:
            info = 5
        if stp == stpmin and (f > ftest1 or dg >= dgtest):
            info = 4
        if nfev >= maxfev:
            info = 3
        if brackt and stmax - stmin <= xtol * stmax:
            info = 2
        if f <= ftest1 and abs(dg) <= gtol * -dginit:
            info = 1

        if info != 0:
            return x, f, g, stp, info, nfev

        if stage1 and f <= ftest1 and dg >= min(ftol, gtol) * dginit:
            stage1 = False

        if stage1 and f <= fx_ls and f > ftest1:
            fm = f - stp * dgtest
            fxm = fx_ls - stx * dgtest
            fym = fy - sty * dgtest
            dgm = dg - dgtest
            dgxm = dgx - dgtest
            dgym = dgy - dgtest
            stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, infoc = _cstep(
                stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax
            )
            fx_ls = fxm + stx * dgtest
            fy = fym + sty * dgtest
            dgx = dgxm + dgtest
            dgy = dgym + dgtest
        else:
            stx, fx_ls, dgx, sty, fy, dgy, stp, f, dg, brackt, infoc = _cstep(
                stx, fx_ls, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax
            )

        if brackt:
            if abs(sty - stx) >= p66 * width1:
                stp = stx + p5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)


# ---------------------------------------------------------------------------
# NCG optimizer
# ---------------------------------------------------------------------------

def ncg(fun, x0, max_iters=3, stop_tol=1e-13, rel_func_tol=1e-10,
        linesearch_maxfev=5, v_bounds=(1300.0, 1700.0)):
    """Nonlinear Conjugate Gradient (Polak-Ribiere) with More-Thuente line search.

    Args:
        fun: callable(x) -> (f, g) objective and gradient
        x0: initial point (slowness)
        max_iters: max NCG iterations
        stop_tol: gradient norm stopping tolerance
        rel_func_tol: relative function change tolerance
        linesearch_maxfev: max function evaluations in line search
        v_bounds: velocity bounds (min, max) in m/s

    Returns:
        x_opt: optimized slowness
    """
    ub = torch.ones_like(x0) * (1.0 / v_bounds[0])
    lb = torch.ones_like(x0) * (1.0 / v_bounds[1])

    xk = x0
    fk, gk = fun(xk)
    nx = xk.numel()

    for it in range(max_iters):
        if it == 0:
            pk = -gk
            ak = 0.01
            gkTgk = torch.dot(gk.flatten(), gk.flatten())
            is_init = True
        else:
            # Polak-Ribiere
            gkTgk = torch.dot(gk.flatten(), gk.flatten())
            gk_diff = gk - gkold
            if gkTgkold > 0:
                bk = torch.dot(gk.flatten(), gk_diff.flatten()) / gkTgkold
            else:
                bk = 0
            if bk < 0:
                bk = max(0, bk)
            pk = -gk + bk * pkold

        gkold = gk
        pkold = pk
        gkTgkold = gkTgk

        # Line search
        if is_init:
            a0 = 1.0
            is_init = False
        else:
            a0 = ak

        xk, fk, gk, ak, lsinfo, nfev = _cvsrch(
            fun, xk, fk, gk, a0, pk, ub, lb, linesearch_maxfev
        )

        if lsinfo != 1:
            logging.info(f"Warning: line search warning = {lsinfo}")

        g2norm = torch.norm(gk)
        g2normnx = g2norm / nx
        logging.info(f"NCG iter {it}: F={fk:.4e}, ||G||/N={g2normnx:.4e}")

        if g2normnx < stop_tol:
            break
        if it > 0:
            relfit = abs(fk - fkold) / max(abs(fkold), torch.finfo(torch.float32).eps)
            if relfit <= rel_func_tol:
                break
        fkold = fk

    return xk


# ---------------------------------------------------------------------------
# Single- and multi-frequency inversion
# ---------------------------------------------------------------------------

def invert_single_frequency(freq_mhz, dobs, sigma, slowness, ix, iy, R,
                            all_u_size, dh=50, ppw=8, lamb=1,
                            boundary_widths=20, born_max=500,
                            energy_threshold=1e-5, ncg_iters=3,
                            v_bounds=(1300.0, 1700.0)):
    """Run FWI for one frequency.

    Args:
        freq_mhz: frequency in MHz (e.g. 0.3)
        dobs: (n_rec, n_rec) complex observation data on CUDA
        sigma: Gaussian blur sigma for gradient
        slowness: (nx, ny) current slowness model on CUDA
        ix, iy: (n_rec,) 1-indexed receiver positions
        R: sparse restriction operator
        all_u_size: [nx, ny, n_rec]
        dh, ppw, lamb, boundary_widths, born_max, energy_threshold: CBS params
        ncg_iters: NCG iterations
        v_bounds: velocity bounds

    Returns:
        updated slowness (nx, ny)
    """
    from src.preprocessing import create_dobs_masks

    freq = freq_mhz * 10  # convert MHz to 100kHz units

    dobs_masked, mask_esi, mask_misfit = create_dobs_masks(
        dobs, ix, iy, dh, mute_dist=7500
    )

    get_grad = create_objective(
        all_u_size, freq, dobs_masked, mask_esi, mask_misfit,
        R, ix, iy, sigma, dh, ppw, lamb, boundary_widths, born_max, energy_threshold
    )

    # Initial scaling
    f0, g0 = get_grad(slowness)
    fscale = f0 / 1e3
    gscale = torch.norm(g0, p=2) / torch.norm(slowness, p=2) * 1e3

    fun = lambda x: get_grad(slowness=x, fscale=fscale, gscale=gscale)

    slowness = ncg(fun, slowness, max_iters=ncg_iters,
                   linesearch_maxfev=5, v_bounds=v_bounds)
    return slowness


def invert_multi_frequency(frequencies_mhz, dobs_dict, slowness, ix, iy, R,
                           all_u_size, dh=50, ppw=8, lamb=1,
                           boundary_widths=20, born_max=500,
                           energy_threshold=1e-5, ncg_iters=3,
                           v_bounds=(1300.0, 1700.0)):
    """Run multi-frequency FWI with frequency bootstrapping.

    Args:
        frequencies_mhz: sorted list of frequencies in MHz
        dobs_dict: {freq_str: dobs_tensor} for each frequency
        Other args same as invert_single_frequency

    Returns:
        final_slowness, history list of (freq, vp_array) tuples
    """
    history = []
    for freq_mhz in frequencies_mhz:
        freq_str = f"{freq_mhz:g}"
        logging.info(f"Beginning Frequency {freq_mhz} MHz")
        dobs = dobs_dict[freq_str]

        # Sigma schedule: low freq -> large sigma, high freq -> small sigma
        if freq_mhz <= 0.3:
            sigma = 5
        elif freq_mhz < 0.8:
            sigma = 2
        else:
            sigma = 1

        slowness = invert_single_frequency(
            freq_mhz, dobs, sigma, slowness, ix, iy, R,
            all_u_size, dh, ppw, lamb, boundary_widths,
            born_max, energy_threshold, ncg_iters, v_bounds
        )

        vp = (1 / slowness).cpu().numpy()
        history.append((freq_mhz, vp.copy()))
        logging.info(f"Frequency {freq_mhz} MHz done. Vp range: [{vp.min():.1f}, {vp.max():.1f}]")

    return slowness, history
