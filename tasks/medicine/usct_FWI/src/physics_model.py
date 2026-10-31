"""CBS (Convergent Born Series) Helmholtz solver for USCT forward modeling.

Solves the Helmholtz equation (nabla^2 + k^2)u = -rho using an iterative
Born series with guaranteed convergence for arbitrarily large scattering
potentials. The method uses "wiggle" phase ramps and FFT-based propagation.

Adapted from CBS_FWI_torch/utils/CBS_utils.py.
"""

import math
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Domain setup helpers
# ---------------------------------------------------------------------------

def ensure_3d_shape(tensor, boundary_widths):
    """Pad a 2D tensor shape to 3D and compute padded domain size with FFT-friendly dims."""
    shape = tensor.shape
    original_dim = len(shape)
    flag_dim = [original_dim >= 1, original_dim >= 2, original_dim == 3]
    three_shape = shape + (1,) * (3 - original_dim)
    bw_size = [boundary_widths if flag else 0 for flag in flag_dim]
    all_size = [a + 2 * b for a, b in zip(three_shape, bw_size)]
    adjusted_all_size = _adjust_sizes(all_size, flag_dim)
    Bl, Br = zip(
        *(
            (math.ceil((a - b) / 2), math.floor((a - b) / 2))
            for a, b in zip(adjusted_all_size, three_shape)
        )
    )
    roi = [list(Bl), list(Br)]
    return adjusted_all_size, roi, bw_size, flag_dim


def _adjust_sizes(all_size, flag_dim):
    """Adjust sizes to be FFT-friendly (small prime factors only)."""
    for i in range(len(all_size)):
        if not flag_dim[i]:
            continue
        item = all_size[i]
        while True:
            factors = list(sympy.factorint(item).keys())
            if max(factors) <= 11 and (len(factors) <= 2 or sorted(factors)[-2] <= 5):
                break
            item += 1
        all_size[i] = item
    return all_size


def _replicate_padding_lb(e_r, Bl):
    """Replicate-pad left/bottom edges."""
    h_bl, w_bl = Bl[0], Bl[1]
    h, w = e_r.shape
    padded = torch.zeros((h + h_bl, w + w_bl)).cuda()
    padded[h_bl:, w_bl:] = e_r
    for i in range(h_bl):
        padded[i, :] = padded[h_bl, :]
    for i in range(w_bl):
        padded[:, i] = padded[:, w_bl]
    return padded


def _replicate_padding_rt(e_r, Br):
    """Replicate-pad right/top edges."""
    h_br, w_br = Br[0], Br[1]
    h, w = e_r.shape
    padded = torch.zeros((h + h_br, w + w_br)).cuda()
    padded[:-h_br, :-w_br] = e_r
    for i in range(h_br):
        padded[-(i + 1), :] = padded[-(h_br + 1), :]
    for i in range(w_br):
        padded[:, -(i + 1)] = padded[:, -(w_br + 1)]
    return padded


def _add_boundary_layer(e_r, roi):
    Bl, Br = roi
    temp = _replicate_padding_lb(e_r, Bl)
    return _replicate_padding_rt(temp, Br)


def _nuttallwin(N, sflag="symmetric"):
    """Nuttall minimum 4-term Blackman-Harris window."""
    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    if sflag == "periodic":
        x = torch.arange(0, N) * 2.0 * torch.pi / N
    else:
        x = torch.arange(0, N) * 2.0 * torch.pi / (N - 1)
    return a[0] - a[1] * torch.cos(x) + a[2] * torch.cos(2 * x) - a[3] * torch.cos(3 * x)


def _apply_edge_filters(V, N, roi, boundary_widths):
    """Apply absorbing boundary edge filters to the potential map."""
    Bl, Br = roi
    for dim in range(len(N)):
        bl = Bl[dim]
        br = Br[dim]
        roi_size = N[dim] - bl - br
        if bl > 0:
            L = boundary_widths[dim]
            if all(n > 1 for n in N):
                window = lambda B: ((torch.arange(1, B + 1) - 0.21) / (B + 0.66))
            else:
                win = lambda nutall, B: nutall[:B]
                window = lambda B: win(_nuttallwin(2 * L - 1), B)

            if br == bl:
                smoothstep = torch.cat([torch.tensor([0.0]), window(L - 1)])
            else:
                smoothstep = window(L)

            filt = torch.cat([
                torch.zeros(bl - L),
                smoothstep,
                torch.ones(roi_size),
                torch.flipud(smoothstep),
                torch.zeros(br - L),
            ]).cuda()

            filter_shape = [1] * V.dim()
            filter_shape[dim] = filt.size(0)
            filt = filt.view(filter_shape)
            V = V * filt
    return V


def _ri2potential(e_r, k0, k0c, adjusted_all_size, roi, bw_size, epsilonmin=3):
    """Convert refractive index to scattering potential."""
    padded_e_r = _add_boundary_layer(e_r, roi)
    V_tot = padded_e_r * k0**2 - k0c**2
    Vabs_max = torch.max(torch.abs(V_tot))
    max_index = torch.argmax(torch.abs(V_tot))
    if torch.real(V_tot.flatten()[max_index]) < 0.05 * Vabs_max:
        Vabs_max *= 1.05
    epsilon = max(Vabs_max.item(), epsilonmin)
    V = padded_e_r * k0**2 - k0c**2 - 1j * epsilon
    V_filtered = _apply_edge_filters(V, adjusted_all_size, roi, bw_size)
    return V_filtered, epsilon


def _expand_dims(tensor, flag_dim):
    for dim, flag in enumerate(flag_dim):
        if not flag:
            tensor = tensor.unsqueeze(dim)
    return tensor


def _build_grid(pixel_size, adjusted_all_size):
    """Build spatial and frequency grids for CBS propagation."""
    x = pixel_size * torch.arange(adjusted_all_size[0]).reshape(-1, 1, 1).cuda()
    y = pixel_size * torch.arange(adjusted_all_size[1]).reshape(1, -1, 1).cuda()
    z = pixel_size * torch.arange(adjusted_all_size[2]).reshape(1, 1, -1).cuda()

    dfx = 2 * torch.pi / (pixel_size * adjusted_all_size[0])
    dfy = 2 * torch.pi / (pixel_size * adjusted_all_size[1])
    dfz = 2 * torch.pi / (pixel_size * adjusted_all_size[2])

    fx = (dfx * torch.fft.fftshift(
        torch.arange(adjusted_all_size[0]) - torch.floor(torch.tensor(adjusted_all_size[0] / 2))
    ).reshape(-1, 1, 1).cuda())

    fy = (dfy * torch.fft.fftshift(
        torch.arange(adjusted_all_size[1]) - torch.floor(torch.tensor(adjusted_all_size[1] / 2))
    ).reshape(1, -1, 1).cuda())

    fz = (dfz * torch.fft.fftshift(
        torch.arange(adjusted_all_size[2]) - torch.floor(torch.tensor(adjusted_all_size[2] / 2))
    ).reshape(1, 1, -1).cuda())

    return {
        "adjusted_all_size": adjusted_all_size,
        "pixel_size": pixel_size,
        "dfx": dfx, "x": x, "fx": fx,
        "dfy": dfy, "y": y, "fy": fy,
        "dfz": dfz, "z": z, "fz": fz,
    }


def _wiggle_perm(wiggle_flags):
    """Generate all wiggle direction permutations."""
    n_directions = len(wiggle_flags)
    n_wiggles = sum(wiggle_flags)
    wiggle_set = torch.zeros((n_directions, 2**n_wiggles))
    wiggle_set[wiggle_flags, :] = (-1) ** (
        torch.ceil(
            torch.arange(1, 2**n_wiggles + 1) / 2 ** torch.arange(n_wiggles).view(-1, 1)
        ) + 1
    )
    return wiggle_set


def _wiggle_descriptor(grid, wig, epsilon, k02e):
    """Compute phase ramps and propagation kernel for one wiggle permutation."""
    sqrt_epsilon = math.sqrt(epsilon)
    pxe = (grid["fx"] - grid["dfx"] * wig[0] / 4) / sqrt_epsilon
    pye = (grid["fy"] - grid["dfy"] * wig[1] / 4) / sqrt_epsilon
    pze = (grid["fz"] - grid["dfz"] * wig[2] / 4) / sqrt_epsilon

    gx = torch.exp(
        (1j * wig[0] * grid["x"]) * (torch.pi / 2) / (grid["pixel_size"] * grid["x"].shape[0])
    ).cuda()
    gy = torch.exp(
        (1j * wig[1] * grid["y"]) * (torch.pi / 2) / (grid["pixel_size"] * grid["y"].shape[1])
    ).cuda()
    gz = torch.exp(
        (1j * wig[2] * grid["z"]) * (torch.pi / 2) / (grid["pixel_size"] * grid["z"].shape[2])
    ).cuda()

    return {
        "filter_step0": gx * gy * gz,
        "scaler_step1": pxe**2 + pye**2 + pze**2 - k02e,
        "filter_step2": gx.conj() * gy.conj() * gz.conj(),
    }


def _compute_wiggles(grid, flag_dim, epsilon, k02e):
    """Compute all wiggle descriptors."""
    wiggle_set = _wiggle_perm(flag_dim)
    n_wiggles = wiggle_set.size(1)
    return [_wiggle_descriptor(grid, wiggle_set[:, w], epsilon, k02e) for w in range(n_wiggles)]


# ---------------------------------------------------------------------------
# Domain setup
# ---------------------------------------------------------------------------

def setup_domain(velocity, freq, dh=50, ppw=8, lamb=1, boundary_widths=20,
                 born_max=500, energy_threshold=1e-5):
    """Prepare the CBS simulation domain from velocity field and frequency.

    Args:
        velocity: float32 tensor (nx, ny) - sound speed field in m/s
        freq: frequency in 100kHz units (e.g. 3.0 for 0.3 MHz)
        dh: grid spacing in um
        ppw: points per wavelength
        lamb: wavelength parameter
        boundary_widths: absorbing boundary width in grid cells
        born_max: max Born iterations
        energy_threshold: convergence threshold

    Returns:
        dict with all precomputed domain quantities needed by cbs_solve
    """
    pixel_size = lamb / ppw
    bw = boundary_widths * ppw
    scale = dh / pixel_size
    speed_mean = torch.mean(velocity)
    lamb_val = speed_mean / (freq * scale)

    refractive_index = speed_mean / velocity
    e_r = refractive_index**2
    e_r_min = torch.min(e_r)
    e_r_max = torch.max(e_r)
    e_r_center = (e_r_max + e_r_min) / 2

    k0 = 2 * torch.pi / lamb_val
    k0c = torch.sqrt(e_r_center) * k0

    adjusted_all_size, roi, bw_size, flag_dim = ensure_3d_shape(e_r, bw)
    grid = _build_grid(pixel_size, adjusted_all_size)
    V, epsilon = _ri2potential(e_r, k0, k0c, adjusted_all_size, roi, bw_size)
    gamma = (1j / epsilon) * V
    gamma_expand = _expand_dims(gamma, flag_dim)
    mix_weight1 = 1 - gamma_expand
    mix_weight2 = 1j * gamma_expand**2

    iterations_per_cycle = lamb_val / (2 * k0c / epsilon)
    k02e = k0c**2 / epsilon + 1j
    wiggles = _compute_wiggles(grid, flag_dim, epsilon, k02e)
    n_wiggle = len(wiggles)

    # Source shift parameters
    Bl, _ = roi
    A = 1j / epsilon / n_wiggle

    return {
        "grid": grid,
        "wiggles": wiggles,
        "n_wiggle": n_wiggle,
        "gamma": gamma_expand.cuda(),
        "mix_weight1": mix_weight1.cuda(),
        "mix_weight2": mix_weight2.cuda(),
        "roi": roi,
        "flag_dim": flag_dim,
        "epsilon": epsilon,
        "k02e": k02e,
        "Bl": Bl,
        "A": A,
        "dh": dh,
        "energy_threshold": energy_threshold,
        "born_max": born_max,
        "max_iterations": math.ceil(born_max * iterations_per_cycle),
    }


# ---------------------------------------------------------------------------
# CBS solve
# ---------------------------------------------------------------------------

def _get_cropped_field_fn(flag_dim, roi):
    """Return a function that crops field to the ROI."""
    Bl, Br = roi
    if flag_dim == [True, True, False]:
        def fn(dE):
            return torch.squeeze(dE[Bl[0]:-Br[0], Bl[1]:-Br[1], :])
        return fn
    elif flag_dim == [True, False, False]:
        def fn(dE):
            return torch.squeeze(dE[Bl[0]:-Br[0], :, :])
        return fn
    elif flag_dim == [True, True, True]:
        def fn(dE):
            return torch.squeeze(dE[Bl[0]:-Br[0], Bl[1]:-Br[1], Bl[2]:-Br[2]])
        return fn


def cbs_solve(ix, iy, domain):
    """Solve Helmholtz equation for a single source using CBS iteration.

    Args:
        ix, iy: 0-indexed source position on the grid
        domain: dict from setup_domain

    Returns:
        scaled_field: complex64 (nx, ny) - the wavefield solution
    """
    Bl = domain["Bl"]
    A = domain["A"]
    wiggles = domain["wiggles"]
    n_wiggle = domain["n_wiggle"]
    mix_weight1 = domain["mix_weight1"]
    mix_weight2 = domain["mix_weight2"]
    gamma = domain["gamma"]
    dh = domain["dh"]
    energy_threshold = domain["energy_threshold"]
    born_max = domain["born_max"]
    crop_fn = _get_cropped_field_fn(domain["flag_dim"], domain["roi"])

    # Shifted source position
    src_x = int(ix) + Bl[0]
    src_y = int(iy) + Bl[1]
    src_z = 0 + Bl[2]
    src_value = 1.0 * A

    adjusted_size = domain["grid"]["adjusted_all_size"]
    dE = torch.zeros(adjusted_size, dtype=torch.complex64).cuda()
    E = dE.clone()
    iteration = 0
    has_next = True
    init_energy = None

    while has_next:
        if iteration < n_wiggle:
            Etmp = dE.clone()
            Etmp[src_x, src_y, src_z] += src_value
        else:
            Etmp = dE.clone()

        idx_w = iteration % n_wiggle
        wig = wiggles[idx_w]

        # CBS propagation step
        E_filtered = Etmp * wig["filter_step0"]
        fftE = torch.fft.fftn(E_filtered)
        fftE_scaled = fftE / wig["scaler_step1"]
        E_ifft = torch.fft.ifftn(fftE_scaled)
        E_return = E_ifft * wig["filter_step2"]
        dE = mix_weight1 * dE - mix_weight2 * E_return

        E += dE

        croped = crop_fn(dE)
        last_energy = torch.sum(torch.abs(croped) ** 2)
        can_terminate = iteration % n_wiggle
        if iteration == 0:
            init_energy = last_energy
        if can_terminate == 0 and (
            last_energy / init_energy < energy_threshold
            or iteration >= born_max
        ):
            has_next = False

        iteration += 1

    E = E / gamma
    E = crop_fn(E)
    scaled_E = -torch.conj(E).T / dh / dh * 2.5e10
    return scaled_E


def solve_all_sources(velocity, ix_arr, iy_arr, freq, dh=50, ppw=8, lamb=1,
                      boundary_widths=20, born_max=500, energy_threshold=1e-5):
    """Solve CBS for all source positions, reusing domain setup.

    Args:
        velocity: (nx, ny) velocity field
        ix_arr, iy_arr: (n_src,) 1-indexed source positions
        freq: frequency in 100kHz units
        dh, ppw, lamb, boundary_widths, born_max, energy_threshold: CBS parameters

    Returns:
        all_fields: complex64 (nx, ny, n_src)
    """
    domain = setup_domain(velocity, freq, dh, ppw, lamb, boundary_widths,
                          born_max, energy_threshold)
    nx, ny = velocity.shape
    n_src = len(ix_arr)
    result = torch.zeros((nx, ny, n_src), dtype=torch.complex64).cuda()

    for i in tqdm(range(n_src)):
        # Convert 1-indexed to 0-indexed
        result[..., i] = cbs_solve(ix_arr[i] - 1, iy_arr[i] - 1, domain)

    return result
