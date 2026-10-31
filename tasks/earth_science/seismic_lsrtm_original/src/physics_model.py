"""2D acoustic wave propagation with C-PML and Born forward modeling.

All deepwave dependencies removed.  The algorithm follows:
  - Pasalic & McGarry (2010) SEG: Convolutional PML for acoustic wave equations.
  - Deepwave Python backend (Alan Richardson): PML profile construction and
    finite-difference time-domain loop logic.
  - Born scattering: linearized (single-scattering) approximation where the
    total wavefield is split into background + scattered components.

Wave equation (2nd-order in time, 4th-order FD in space):
    p^{n+1} = v^2 * dt^2 * W_pml(p^n, psi^n, zeta^n) + 2*p^n - p^{n-1}

Born scattered wavefield update:
    p_sc^{n+1} = v^2 * dt^2 * W_pml(p_sc^n, ...) + 2*p_sc^n - p_sc^{n-1}
                 + 2*v*scatter*dt^2 * W_pml(p_bg^n, ...)

where W_pml is the PML-modified Laplacian using auxiliary variables psi, zeta.

CFL condition: inner_dt = dt / step_ratio, step_ratio = ceil(dt / dt_max),
    dt_max = 0.6 / (sqrt(1/dy^2 + 1/dx^2) * v_max).

Source amplitude is pre-scaled by -v(xs)^2 * inner_dt^2 and injected after
each wave step into the BACKGROUND wavefield only.

Receiver data is recorded from the SCATTERED wavefield.
"""

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


# ---------------------------------------------------------------------------
# 4th-order finite-difference stencils
# ---------------------------------------------------------------------------

def _fd1_y(a: torch.Tensor, rdy: float) -> torch.Tensor:
    """4th-order central first derivative along y (dim -2)."""
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1, :] - a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] - a[..., :-4, :])) * rdy,
        (0, 0, 2, 2),
    )


def _fd1_x(a: torch.Tensor, rdx: float) -> torch.Tensor:
    """4th-order central first derivative along x (dim -1)."""
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1] - a[..., 1:-3])
         - 1.0 / 12.0 * (a[..., 4:] - a[..., :-4])) * rdx,
        (2, 2),
    )


def _fd2_y(a: torch.Tensor, rdy2: float) -> torch.Tensor:
    """4th-order central second derivative along y (dim -2)."""
    return F.pad(
        (-5.0 / 2.0 * a[..., 2:-2, :]
         + 4.0 / 3.0 * (a[..., 3:-1, :] + a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] + a[..., :-4, :])) * rdy2,
        (0, 0, 2, 2),
    )


def _fd2_x(a: torch.Tensor, rdx2: float) -> torch.Tensor:
    """4th-order central second derivative along x (dim -1)."""
    return F.pad(
        (-5.0 / 2.0 * a[..., 2:-2]
         + 4.0 / 3.0 * (a[..., 3:-1] + a[..., 1:-3])
         - 1.0 / 12.0 * (a[..., 4:] + a[..., :-4])) * rdx2,
        (2, 2),
    )


# ---------------------------------------------------------------------------
# CFL condition
# ---------------------------------------------------------------------------

def cfl_step_ratio(
    dy: float,
    dx: float,
    dt: float,
    v_max: float,
    c_max: float = 0.6,
) -> Tuple[float, int]:
    """Compute CFL-stable inner time step and step ratio.

    Args:
        dy: Grid spacing in y (meters).
        dx: Grid spacing in x (meters).
        dt: User-level time step (seconds).
        v_max: Maximum wave speed in model (m/s).
        c_max: Maximum Courant number. Default 0.6 (matches deepwave).

    Returns:
        inner_dt: Inner time step satisfying CFL condition.
        step_ratio: Integer n such that inner_dt = dt / n.
    """
    max_dt = c_max / math.sqrt(1.0 / dy**2 + 1.0 / dx**2) / v_max
    step_ratio = math.ceil(abs(dt) / max_dt)
    inner_dt = dt / step_ratio
    return inner_dt, step_ratio


# ---------------------------------------------------------------------------
# C-PML profile construction
# ---------------------------------------------------------------------------

def _setup_pml_1d(
    n: int,
    pml_start_left: float,
    pml_start_right: float,
    pml_width: int,
    sigma0: float,
    alpha0: float,
    inner_dt: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 1D C-PML a and b profiles for one spatial dimension."""
    n_power = 2
    eps = 1e-9
    x = torch.arange(n, device=device, dtype=dtype)

    pml_frac_left = (pml_start_left - x) / (pml_width + eps)
    pml_frac_right = (x - pml_start_right) / (pml_width + eps)
    pml_frac = torch.clamp(torch.maximum(pml_frac_left, pml_frac_right), 0.0, 1.0)

    sigma = sigma0 * pml_frac ** n_power
    alpha = alpha0 * (1.0 - pml_frac)
    sigmaalpha = sigma + alpha

    a = torch.exp(-sigmaalpha * abs(inner_dt))
    b = sigma / sigmaalpha * (a - 1.0)
    a[pml_frac == 0] = 0.0
    return a, b


def setup_pml_profiles(
    ny_p: int,
    nx_p: int,
    pml_width: int,
    fd_pad: int,
    dy: float,
    dx: float,
    inner_dt: float,
    v_max: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    r_val: float = 0.001,
    n_power: int = 2,
) -> List[torch.Tensor]:
    """Construct 2D C-PML profiles for all sides.

    Returns a list [ay, by, dbydy, ax, bx, dbxdx].
    """
    alpha0 = math.pi * pml_freq
    max_pml = pml_width * max(dy, dx)

    if max_pml == 0 or pml_width == 0:
        zeros_y = torch.zeros(ny_p, 1, device=device, dtype=dtype)
        zeros_x = torch.zeros(1, nx_p, device=device, dtype=dtype)
        return [zeros_y, zeros_y, zeros_y, zeros_x, zeros_x, zeros_x]

    sigma0 = -(1 + n_power) * v_max * math.log(r_val) / (2.0 * max_pml)

    interior_start = fd_pad + pml_width
    interior_end_y = ny_p - 1 - fd_pad - pml_width
    interior_end_x = nx_p - 1 - fd_pad - pml_width

    ay_1d, by_1d = _setup_pml_1d(
        ny_p, interior_start, interior_end_y,
        pml_width, sigma0, alpha0, inner_dt, dtype, device,
    )
    dbydy_1d = _fd1_x(by_1d.unsqueeze(0), 1.0 / dy).squeeze(0)

    ax_1d, bx_1d = _setup_pml_1d(
        nx_p, interior_start, interior_end_x,
        pml_width, sigma0, alpha0, inner_dt, dtype, device,
    )
    dbxdx_1d = _fd1_x(bx_1d.unsqueeze(0), 1.0 / dx).squeeze(0)

    ay = ay_1d.reshape(ny_p, 1)
    by = by_1d.reshape(ny_p, 1)
    dbydy = dbydy_1d.reshape(ny_p, 1)
    ax = ax_1d.reshape(1, nx_p)
    bx = bx_1d.reshape(1, nx_p)
    dbxdx = dbxdx_1d.reshape(1, nx_p)

    return [ay, by, dbydy, ax, bx, dbxdx]


# ---------------------------------------------------------------------------
# Single C-PML wave step
# ---------------------------------------------------------------------------

def wave_step(
    v_p: torch.Tensor,
    wfc: torch.Tensor,
    wfp: torch.Tensor,
    psi_y: torch.Tensor,
    psi_x: torch.Tensor,
    zeta_y: torch.Tensor,
    zeta_x: torch.Tensor,
    pml_profiles: List[torch.Tensor],
    dy: float,
    dx: float,
    inner_dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single C-PML time step for 2D scalar acoustic wave equation.

    Returns:
        Tuple (new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x).
    """
    ay, by, dbydy, ax, bx, dbxdx = pml_profiles

    rdy = 1.0 / dy
    rdx = 1.0 / dx
    rdy2 = rdy * rdy
    rdx2 = rdx * rdx

    dwfc_dy = _fd1_y(wfc, rdy)
    dwfc_dx = _fd1_x(wfc, rdx)

    tmp_y = (
        (1.0 + by) * _fd2_y(wfc, rdy2)
        + dbydy * dwfc_dy
        + _fd1_y(ay * psi_y, rdy)
    )
    tmp_x = (
        (1.0 + bx) * _fd2_x(wfc, rdx2)
        + dbxdx * dwfc_dx
        + _fd1_x(ax * psi_x, rdx)
    )

    w_sum = (1.0 + by) * tmp_y + ay * zeta_y + (1.0 + bx) * tmp_x + ax * zeta_x

    new_wfc = v_p ** 2 * inner_dt ** 2 * w_sum + 2.0 * wfc - wfp

    new_psi_y = by * dwfc_dy + ay * psi_y
    new_psi_x = bx * dwfc_dx + ax * psi_x
    new_zeta_y = by * tmp_y + ay * zeta_y
    new_zeta_x = bx * tmp_x + ax * zeta_x

    return new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x


# ---------------------------------------------------------------------------
# Born wave step: background + scattered wavefield
# ---------------------------------------------------------------------------

def born_wave_step(
    v_p: torch.Tensor,
    wfc: torch.Tensor,
    wfp: torch.Tensor,
    psi_y: torch.Tensor,
    psi_x: torch.Tensor,
    zeta_y: torch.Tensor,
    zeta_x: torch.Tensor,
    wfc_sc: torch.Tensor,
    wfp_sc: torch.Tensor,
    psi_y_sc: torch.Tensor,
    psi_x_sc: torch.Tensor,
    zeta_y_sc: torch.Tensor,
    zeta_x_sc: torch.Tensor,
    scatter_p: torch.Tensor,
    pml_profiles: List[torch.Tensor],
    dy: float,
    dx: float,
    inner_dt: float,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Single C-PML time step for both background and Born-scattered wavefields.

    The background wavefield evolves identically to ``wave_step``.
    The scattered wavefield is driven by the Born coupling term:

        p_sc^{n+1} = v^2 * dt^2 * W_sc + 2*p_sc^n - p_sc^{n-1}
                     + 2*v*scatter*dt^2 * W_bg

    where W_bg and W_sc are PML-modified Laplacians of the background and
    scattered wavefields respectively.

    Args:
        v_p: Padded velocity model, shape (ny_p, nx_p).
        wfc, wfp: Background current/previous pressure.
        psi_y, psi_x, zeta_y, zeta_x: Background PML auxiliaries.
        wfc_sc, wfp_sc: Scattered current/previous pressure.
        psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc: Scattered PML auxiliaries.
        scatter_p: Padded scattering potential, shape (ny_p, nx_p).
        pml_profiles: [ay, by, dbydy, ax, bx, dbxdx].
        dy, dx: Grid spacings.
        inner_dt: Internal time step.

    Returns:
        Tuple of 10 tensors:
        (new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x,
         new_wfc_sc, new_psi_y_sc, new_psi_x_sc, new_zeta_y_sc, new_zeta_x_sc)
    """
    ay, by, dbydy, ax, bx, dbxdx = pml_profiles

    rdy = 1.0 / dy
    rdx = 1.0 / dx
    rdy2 = rdy * rdy
    rdx2 = rdx * rdx

    # --- Background wavefield ---
    dwfc_dy = _fd1_y(wfc, rdy)
    dwfc_dx = _fd1_x(wfc, rdx)

    tmp_y = (
        (1.0 + by) * _fd2_y(wfc, rdy2)
        + dbydy * dwfc_dy
        + _fd1_y(ay * psi_y, rdy)
    )
    tmp_x = (
        (1.0 + bx) * _fd2_x(wfc, rdx2)
        + dbxdx * dwfc_dx
        + _fd1_x(ax * psi_x, rdx)
    )

    w_sum = (1.0 + by) * tmp_y + ay * zeta_y + (1.0 + bx) * tmp_x + ax * zeta_x
    new_wfc = v_p ** 2 * inner_dt ** 2 * w_sum + 2.0 * wfc - wfp

    new_psi_y = by * dwfc_dy + ay * psi_y
    new_psi_x = bx * dwfc_dx + ax * psi_x
    new_zeta_y = by * tmp_y + ay * zeta_y
    new_zeta_x = bx * tmp_x + ax * zeta_x

    # --- Scattered wavefield ---
    dwfc_sc_dy = _fd1_y(wfc_sc, rdy)
    dwfc_sc_dx = _fd1_x(wfc_sc, rdx)

    tmp_y_sc = (
        (1.0 + by) * _fd2_y(wfc_sc, rdy2)
        + dbydy * dwfc_sc_dy
        + _fd1_y(ay * psi_y_sc, rdy)
    )
    tmp_x_sc = (
        (1.0 + bx) * _fd2_x(wfc_sc, rdx2)
        + dbxdx * dwfc_sc_dx
        + _fd1_x(ax * psi_x_sc, rdx)
    )

    wsc_sum = (1.0 + by) * tmp_y_sc + ay * zeta_y_sc + (1.0 + bx) * tmp_x_sc + ax * zeta_x_sc

    # Born coupling: scattered wavefield driven by background Laplacian
    new_wfc_sc = (
        v_p ** 2 * inner_dt ** 2 * wsc_sum
        + 2.0 * wfc_sc - wfp_sc
        + 2.0 * v_p * scatter_p * inner_dt ** 2 * w_sum
    )

    new_psi_y_sc = by * dwfc_sc_dy + ay * psi_y_sc
    new_psi_x_sc = bx * dwfc_sc_dx + ax * psi_x_sc
    new_zeta_y_sc = by * tmp_y_sc + ay * zeta_y_sc
    new_zeta_x_sc = bx * tmp_x_sc + ax * zeta_x_sc

    return (
        new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x,
        new_wfc_sc, new_psi_y_sc, new_psi_x_sc, new_zeta_y_sc, new_zeta_x_sc,
    )


# Compiled versions for GPU acceleration
try:
    _wave_step_compiled = torch.compile(wave_step)
except Exception:
    _wave_step_compiled = wave_step

try:
    _born_wave_step_compiled = torch.compile(born_wave_step)
except Exception:
    _born_wave_step_compiled = born_wave_step


# ---------------------------------------------------------------------------
# FFT resampling (matches deepwave upsample / downsample)
# ---------------------------------------------------------------------------

def _fft_upsample(signal: torch.Tensor, step_ratio: int) -> torch.Tensor:
    """FFT-based low-pass upsampling of the last dimension by step_ratio."""
    if step_ratio == 1:
        return signal
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    sig_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)
    if sig_f.shape[-1] > 1:
        sig_f = sig_f.clone()
        sig_f[..., -1] = 0
    pad_len = up_nt // 2 + 1 - sig_f.shape[-1]
    if pad_len > 0:
        sig_f = F.pad(sig_f, (0, pad_len))
    return torch.fft.irfft(sig_f, n=up_nt, norm="ortho")


def _fft_downsample(signal: torch.Tensor, step_ratio: int) -> torch.Tensor:
    """FFT-based anti-aliased downsampling of the last dimension by step_ratio."""
    if step_ratio == 1:
        return signal
    nt = signal.shape[-1]
    down_nt = nt // step_ratio
    sig_f = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]
    if sig_f.shape[-1] > 1:
        sig_f = sig_f.clone()
        sig_f[..., -1] = 0
    return torch.fft.irfft(sig_f, n=down_nt, norm="ortho") / math.sqrt(step_ratio)


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def _loc_to_flat(
    loc: torch.Tensor,
    pad_y: int,
    pad_x: int,
    nx_p: int,
) -> torch.Tensor:
    """Convert location indices in original model coordinates to flat indices
    in the padded model.

    Args:
        loc: Shape (..., 2). Last dim is [dim0_idx, dim1_idx].
        pad_y: Padding along dim0.
        pad_x: Padding along dim1.
        nx_p: Width of padded model.

    Returns:
        flat_idx: Shape (...), long tensor.
    """
    dim0_idx = loc[..., 0].long()
    dim1_idx = loc[..., 1].long()
    return ((dim0_idx + pad_y) * nx_p + (dim1_idx + pad_x)).long()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_acquisition_geometry(
    n_shots: int = 20,
    d_source: int = 20,
    first_source: int = 10,
    source_depth: int = 2,
    n_receivers: int = 100,
    d_receiver: int = 6,
    first_receiver: int = 0,
    receiver_depth: int = 2,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """Create source and receiver location tensors for a surface acquisition.

    Uses fixed-spacing geometry matching the LSRTM example.

    Args:
        n_shots: Number of shots.
        d_source: Source spacing in grid cells.
        first_source: First source position along dim0.
        source_depth: Source depth index (dim1).
        n_receivers: Number of receivers per shot.
        d_receiver: Receiver spacing in grid cells.
        first_receiver: First receiver position along dim0.
        receiver_depth: Receiver depth index (dim1).
        device: PyTorch device.

    Returns:
        source_loc: Shape (n_shots, 1, 2) int64, [dim0_idx, dim1_idx].
        receiver_loc: Shape (n_shots, n_receivers, 2) int64.
    """
    source_loc = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_loc[:, 0, 0] = torch.arange(n_shots, device=device) * d_source + first_source
    source_loc[..., 1] = source_depth

    receiver_loc = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    rec_positions = torch.arange(n_receivers, device=device) * d_receiver + first_receiver
    receiver_loc[:, :, 0] = rec_positions.unsqueeze(0).expand(n_shots, -1)
    receiver_loc[..., 1] = receiver_depth

    return source_loc, receiver_loc


def make_ricker_wavelet(
    freq: float,
    nt: int,
    dt: float,
    n_shots: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate Ricker (Mexican-hat) wavelet source amplitudes for all shots.

    Formula: s(t) = (1 - 2*pi^2*f^2*(t-tp)^2) * exp(-pi^2*f^2*(t-tp)^2)
    Peak time tp = 1.5 / freq.

    Args:
        freq: Central frequency in Hz.
        nt: Number of time samples.
        dt: User-level time step in seconds.
        n_shots: Number of shots.
        device: PyTorch device.

    Returns:
        source_amp: Shape (n_shots, 1, nt) float32 tensor.
    """
    peak_time = 1.5 / freq
    t = torch.arange(nt, dtype=torch.float32, device=device) * dt - peak_time
    wavelet = (1.0 - 2.0 * math.pi ** 2 * freq ** 2 * t ** 2) * torch.exp(
        -math.pi ** 2 * freq ** 2 * t ** 2
    )
    return wavelet.unsqueeze(0).unsqueeze(0).expand(n_shots, 1, -1).contiguous()


def forward_model(
    v: torch.Tensor,
    dx: float,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    freq: float,
    accuracy: int = 4,
    pml_width: int = 20,
    checkpoint_every: int = 64,
    max_vel: Optional[float] = None,
) -> torch.Tensor:
    """Simulate acoustic wave propagation and record receiver data.

    Args:
        v: Velocity model, shape (ny, nx), in m/s. May require grad.
        dx: Grid spacing in meters (scalar, same for both dimensions).
        dt: User-level time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt_user).
        source_loc: Source positions, shape (n_shots, 1, 2).
        receiver_loc: Receiver positions, shape (n_shots, n_rec, 2).
        freq: Dominant source frequency (Hz), used for PML tuning.
        accuracy: FD order (only 4 supported). Default 4.
        pml_width: PML width in grid cells on each side. Default 20.
        checkpoint_every: Steps per gradient-checkpoint segment. Default 64.
        max_vel: Optional maximum velocity override for CFL. If None, uses max of v.

    Returns:
        receiver_data: Recorded pressure at receivers, shape (n_shots, n_rec, nt_user).
    """
    assert accuracy == 4, "Only accuracy=4 (4th-order FD) is implemented."

    dy = float(dx)
    dx_f = float(dx)
    ny, nx = v.shape
    fd_pad = accuracy // 2

    pad = fd_pad + pml_width
    ny_p = ny + 2 * pad
    nx_p = nx + 2 * pad

    device = v.device
    dtype = v.dtype

    v_max = float(max_vel) if max_vel is not None else float(v.detach().abs().max().item())
    inner_dt, step_ratio = cfl_step_ratio(dy, dx_f, dt, v_max)

    v_p = F.pad(
        v.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad),
        mode="replicate",
    ).squeeze(0).squeeze(0)

    pml_profiles = setup_pml_profiles(
        ny_p, nx_p, pml_width, fd_pad,
        dy, dx_f, inner_dt, v_max,
        dtype, device, freq,
    )

    nt_user = source_amp.shape[2]
    nt_inner = nt_user * step_ratio

    src_up = _fft_upsample(source_amp, step_ratio)

    src_flat = _loc_to_flat(source_loc, pad, pad, nx_p)
    rec_flat = _loc_to_flat(receiver_loc, pad, pad, nx_p)

    n_shots = source_amp.shape[0]
    flat_size = ny_p * nx_p

    _step_fn = _wave_step_compiled if device.type == "cuda" else wave_step

    v_xs = v_p.reshape(-1)[src_flat[:, 0]]
    src_scaled = -src_up[:, 0, :] * (v_xs ** 2).unsqueeze(1) * (inner_dt ** 2)

    src_indicator = torch.zeros(n_shots, flat_size, device=device, dtype=dtype)
    src_indicator.scatter_(1, src_flat, 1.0)
    src_indicator = src_indicator.view(n_shots, ny_p, nx_p)

    shot_idx = torch.arange(n_shots, device=device)

    wfc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    wfp = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)

    def _make_segment(t_start: int, k: int):
        def _segment(wfc, wfp, psi_y, psi_x, zeta_y, zeta_x):
            rec_list: List[torch.Tensor] = []
            for i in range(k):
                t = t_start + i
                rec_list.append(
                    wfc.reshape(n_shots, -1)[shot_idx.unsqueeze(1), rec_flat]
                )
                wfc_new, psi_y, psi_x, zeta_y, zeta_x = _step_fn(
                    v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                    pml_profiles, dy, dx_f, inner_dt,
                )
                wfc_new = wfc_new + src_scaled[:, t].reshape(n_shots, 1, 1) * src_indicator
                wfp = wfc
                wfc = wfc_new
            rec = torch.stack(rec_list, dim=0)
            return wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec
        return _segment

    rec_segments: List[torch.Tensor] = []

    for seg_start in range(0, nt_inner, checkpoint_every):
        k = min(checkpoint_every, nt_inner - seg_start)
        seg_fn = _make_segment(seg_start, k)

        if v.requires_grad:
            wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec = _checkpoint(
                seg_fn, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                use_reentrant=False,
            )
        else:
            with torch.no_grad():
                wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec = seg_fn(
                    wfc, wfp, psi_y, psi_x, zeta_y, zeta_x
                )

        rec_segments.append(rec)

    rec_tensor = torch.cat(rec_segments, dim=0).permute(1, 2, 0).contiguous()
    return _fft_downsample(rec_tensor, step_ratio)


def born_forward_model(
    v_mig: torch.Tensor,
    scatter: torch.Tensor,
    dx: float,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    freq: float,
    accuracy: int = 4,
    pml_width: int = 20,
    checkpoint_every: int = 64,
    max_vel: Optional[float] = None,
) -> torch.Tensor:
    """Born forward modeling: propagate background + scattered wavefields.

    The background wavefield is driven by the source. The scattered wavefield
    is driven only by the Born coupling term (2*v*scatter*dt^2 * Laplacian_bg).
    Receiver data is recorded from the scattered wavefield.

    Args:
        v_mig: Background (migration) velocity model, shape (ny, nx), m/s.
        scatter: Scattering potential (reflectivity model), shape (ny, nx).
            When scatter.requires_grad is True, gradient checkpointing is used.
        dx: Grid spacing in meters (scalar).
        dt: User-level time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt_user).
        source_loc: Source positions, shape (n_shots, 1, 2).
        receiver_loc: Receiver positions, shape (n_shots, n_rec, 2).
        freq: Dominant source frequency (Hz), for PML tuning.
        accuracy: FD order (only 4). Default 4.
        pml_width: PML width in grid cells. Default 20.
        checkpoint_every: Steps per checkpoint segment. Default 64.
        max_vel: Optional max velocity override for CFL.

    Returns:
        receiver_data: Recorded scattered pressure at receivers,
            shape (n_shots, n_rec, nt_user).
    """
    assert accuracy == 4, "Only accuracy=4 (4th-order FD) is implemented."

    dy = float(dx)
    dx_f = float(dx)
    ny, nx = v_mig.shape
    fd_pad = accuracy // 2

    pad = fd_pad + pml_width
    ny_p = ny + 2 * pad
    nx_p = nx + 2 * pad

    device = v_mig.device
    dtype = v_mig.dtype

    v_max = float(max_vel) if max_vel is not None else float(v_mig.detach().abs().max().item())
    inner_dt, step_ratio = cfl_step_ratio(dy, dx_f, dt, v_max)

    # Pad velocity
    v_p = F.pad(
        v_mig.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad),
        mode="replicate",
    ).squeeze(0).squeeze(0)

    # Pad scatter with zeros (no scattering in PML region)
    scatter_p = F.pad(
        scatter.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad),
        mode="constant",
        value=0.0,
    ).squeeze(0).squeeze(0)

    pml_profiles = setup_pml_profiles(
        ny_p, nx_p, pml_width, fd_pad,
        dy, dx_f, inner_dt, v_max,
        dtype, device, freq,
    )

    nt_user = source_amp.shape[2]
    nt_inner = nt_user * step_ratio

    src_up = _fft_upsample(source_amp, step_ratio)

    src_flat = _loc_to_flat(source_loc, pad, pad, nx_p)
    rec_flat = _loc_to_flat(receiver_loc, pad, pad, nx_p)

    n_shots = source_amp.shape[0]
    flat_size = ny_p * nx_p

    _step_fn = _born_wave_step_compiled if device.type == "cuda" else born_wave_step

    # Source scaling for background wavefield: -v(xs)^2 * inner_dt^2
    v_xs = v_p.reshape(-1)[src_flat[:, 0]]
    src_scaled = -src_up[:, 0, :] * (v_xs ** 2).unsqueeze(1) * (inner_dt ** 2)

    # Source scaling for scattered wavefield: -2*v(xs)*scatter(xs) * inner_dt^2
    scatter_xs = scatter_p.reshape(-1)[src_flat[:, 0]]
    src_scaled_sc = -2.0 * src_up[:, 0, :] * (v_xs * scatter_xs).unsqueeze(1) * (inner_dt ** 2)

    src_indicator = torch.zeros(n_shots, flat_size, device=device, dtype=dtype)
    src_indicator.scatter_(1, src_flat, 1.0)
    src_indicator = src_indicator.view(n_shots, ny_p, nx_p)

    shot_idx = torch.arange(n_shots, device=device)

    # Background wavefields
    wfc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    wfp = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)

    # Scattered wavefields
    wfc_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    wfp_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_y_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_x_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_y_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_x_sc = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)

    requires_grad = scatter.requires_grad or v_mig.requires_grad

    def _make_segment(t_start: int, k: int):
        def _segment(
            wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
            wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
        ):
            rec_list: List[torch.Tensor] = []
            for i in range(k):
                t = t_start + i
                # Record from SCATTERED wavefield
                rec_list.append(
                    wfc_sc.reshape(n_shots, -1)[shot_idx.unsqueeze(1), rec_flat]
                )
                # Born wave step: background + scattered
                (
                    wfc_new, psi_y, psi_x, zeta_y, zeta_x,
                    wfc_sc_new, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                ) = _step_fn(
                    v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                    wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                    scatter_p, pml_profiles, dy, dx_f, inner_dt,
                )
                # Inject source into BACKGROUND wavefield
                wfc_new = wfc_new + src_scaled[:, t].reshape(n_shots, 1, 1) * src_indicator
                # Inject source into SCATTERED wavefield (Born scattering of source)
                wfc_sc_new = wfc_sc_new + src_scaled_sc[:, t].reshape(n_shots, 1, 1) * src_indicator
                wfp = wfc
                wfc = wfc_new
                wfp_sc = wfc_sc
                wfc_sc = wfc_sc_new

            rec = torch.stack(rec_list, dim=0)
            return (
                wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                rec,
            )
        return _segment

    rec_segments: List[torch.Tensor] = []

    for seg_start in range(0, nt_inner, checkpoint_every):
        k = min(checkpoint_every, nt_inner - seg_start)
        seg_fn = _make_segment(seg_start, k)

        if requires_grad:
            (
                wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                rec,
            ) = _checkpoint(
                seg_fn,
                wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                use_reentrant=False,
            )
        else:
            with torch.no_grad():
                (
                    wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                    wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                    rec,
                ) = seg_fn(
                    wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                    wfc_sc, wfp_sc, psi_y_sc, psi_x_sc, zeta_y_sc, zeta_x_sc,
                )

        rec_segments.append(rec)

    rec_tensor = torch.cat(rec_segments, dim=0).permute(1, 2, 0).contiguous()
    return _fft_downsample(rec_tensor, step_ratio)
