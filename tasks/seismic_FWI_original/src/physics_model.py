"""2D acoustic wave propagation with C-PML, implemented from scratch.

All deepwave dependencies removed.  The algorithm follows:
  - Pasalic & McGarry (2010) SEG: Convolutional PML for acoustic wave equations.
  - Deepwave Python backend (Alan Richardson): PML profile construction and
    finite-difference time-domain loop logic.

Wave equation (2nd-order in time, 4th-order FD in space):
    p^{n+1} = v^2 * dt^2 * W_pml(p^n, psi^n, zeta^n) + 2*p^n - p^{n-1}

where W_pml is the PML-modified Laplacian using auxiliary variables psi, zeta.

CFL condition: inner_dt = dt / step_ratio, step_ratio = ceil(dt / dt_max),
    dt_max = 0.6 / (sqrt(1/dy^2 + 1/dx^2) * v_max).

Source amplitude is pre-scaled by -v(xs)^2 * inner_dt^2 and injected after
each wave step, consistent with the deepwave sign convention.

Receiver data recorded before each wave step, then FFT-downsampled back to
the user time step.
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
    """4th-order central first derivative along y (dim -2).

    Stencil applied to interior points j=2..N-3; boundary points zeroed.
    """
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1, :] - a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] - a[..., :-4, :])) * rdy,
        (0, 0, 2, 2),
    )


def _fd1_x(a: torch.Tensor, rdx: float) -> torch.Tensor:
    """4th-order central first derivative along x (dim -1).

    Stencil applied to interior points j=2..N-3; boundary points zeroed.
    """
    return F.pad(
        (8.0 / 12.0 * (a[..., 3:-1] - a[..., 1:-3])
         - 1.0 / 12.0 * (a[..., 4:] - a[..., :-4])) * rdx,
        (2, 2),
    )


def _fd2_y(a: torch.Tensor, rdy2: float) -> torch.Tensor:
    """4th-order central second derivative along y (dim -2).

    Stencil: -5/2*a[i] + 4/3*(a[i+1]+a[i-1]) - 1/12*(a[i+2]+a[i-2])
    """
    return F.pad(
        (-5.0 / 2.0 * a[..., 2:-2, :]
         + 4.0 / 3.0 * (a[..., 3:-1, :] + a[..., 1:-3, :])
         - 1.0 / 12.0 * (a[..., 4:, :] + a[..., :-4, :])) * rdy2,
        (0, 0, 2, 2),
    )


def _fd2_x(a: torch.Tensor, rdx2: float) -> torch.Tensor:
    """4th-order central second derivative along x (dim -1).

    Stencil: -5/2*a[i] + 4/3*(a[i+1]+a[i-1]) - 1/12*(a[i+2]+a[i-2])
    """
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
    """Build 1D C-PML a and b profiles for one spatial dimension.

    Args:
        n: Total grid points including PML and fd_pad.
        pml_start_left: Grid index where the left PML region ends (interior start).
        pml_start_right: Grid index where the right PML region begins (interior end).
        pml_width: PML width in grid cells (same on both sides).
        sigma0: Damping coefficient scale.
        alpha0: Auxiliary alpha0 = pi * pml_freq.
        inner_dt: Internal time step.
        dtype, device: PyTorch tensor properties.

    Returns:
        a: Exponential decay profile, shape (n,).  Zero in interior.
        b: Convolution coefficient profile, shape (n,).
    """
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

    Returns a list [ay, by, dbydy, ax, bx, dbxdx] where:
      - ay, by, dbydy have shape (ny_p, 1) — broadcasts over x.
      - ax, bx, dbxdx have shape (1, nx_p) — broadcasts over y.

    Args:
        ny_p: Total padded grid size in y.
        nx_p: Total padded grid size in x.
        pml_width: PML width in cells (same on all 4 sides).
        fd_pad: Finite-difference padding (accuracy // 2).
        dy, dx: Grid spacings in meters.
        inner_dt: Internal time step.
        v_max: Maximum wave speed.
        dtype, device: PyTorch tensor properties.
        pml_freq: Dominant frequency for PML tuning (Hz).
        r_val: Target reflection coefficient. Default 0.001.
        n_power: Damping profile polynomial order. Default 2.
    """
    alpha0 = math.pi * pml_freq
    max_pml = pml_width * max(dy, dx)

    if max_pml == 0 or pml_width == 0:
        zeros_y = torch.zeros(ny_p, 1, device=device, dtype=dtype)
        zeros_x = torch.zeros(1, nx_p, device=device, dtype=dtype)
        return [zeros_y, zeros_y, zeros_y, zeros_x, zeros_x, zeros_x]

    sigma0 = -(1 + n_power) * v_max * math.log(r_val) / (2.0 * max_pml)

    # Interior region boundaries (grid-index space, 0-based)
    interior_start = fd_pad + pml_width        # first interior index
    interior_end_y = ny_p - 1 - fd_pad - pml_width   # last interior index in y
    interior_end_x = nx_p - 1 - fd_pad - pml_width   # last interior index in x

    # Y dimension profiles
    ay_1d, by_1d = _setup_pml_1d(
        ny_p, interior_start, interior_end_y,
        pml_width, sigma0, alpha0, inner_dt, dtype, device,
    )
    # Derivative of by along y using the same 4th-order FD stencil
    dbydy_1d = _fd1_x(by_1d.unsqueeze(0), 1.0 / dy).squeeze(0)

    # X dimension profiles
    ax_1d, bx_1d = _setup_pml_1d(
        nx_p, interior_start, interior_end_x,
        pml_width, sigma0, alpha0, inner_dt, dtype, device,
    )
    dbxdx_1d = _fd1_x(bx_1d.unsqueeze(0), 1.0 / dx).squeeze(0)

    # Reshape for broadcasting with 2D wavefield (ny_p, nx_p)
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

    Implements the Pasalic & McGarry (2010) convolutional PML update:

        tmp_d = (1+b_d)*∂²p/∂d² + (∂b_d/∂d)*∂p/∂d + ∂(a_d*ψ_d)/∂d
        W = Σ_d [(1+b_d)*tmp_d + a_d*ζ_d]
        p^{n+1} = v^2 * dt^2 * W + 2*p^n - p^{n-1}
        ψ_d^{n+1} = b_d * ∂p/∂d + a_d * ψ_d
        ζ_d^{n+1} = b_d * tmp_d + a_d * ζ_d

    In the interior (a=0, b=0): reduces to the standard 2nd-order Verlet
    time integration with 4th-order FD Laplacian.

    Args:
        v_p: Padded velocity model, shape (ny_p, nx_p).
        wfc: Current pressure wavefield, shape (ny_p, nx_p).
        wfp: Previous pressure wavefield, shape (ny_p, nx_p).
        psi_y, psi_x: PML auxiliary variables (1st kind), same shape as wfc.
        zeta_y, zeta_x: PML auxiliary variables (2nd kind), same shape as wfc.
        pml_profiles: [ay, by, dbydy, ax, bx, dbxdx] from setup_pml_profiles.
        dy, dx: Grid spacings in meters.
        inner_dt: Internal time step (seconds).

    Returns:
        Tuple (new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x).
    """
    ay, by, dbydy, ax, bx, dbxdx = pml_profiles

    rdy = 1.0 / dy
    rdx = 1.0 / dx
    rdy2 = rdy * rdy
    rdx2 = rdx * rdx

    # First derivatives of wfc
    dwfc_dy = _fd1_y(wfc, rdy)
    dwfc_dx = _fd1_x(wfc, rdx)

    # PML-modified second-derivative terms for each dimension
    # tmp_d = (1+b_d)*d2(wfc)/dd2 + (db_d/dd)*d1(wfc)/dd + d1(a_d*psi_d)/dd
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

    # PML-modified Laplacian
    w_sum = (1.0 + by) * tmp_y + ay * zeta_y + (1.0 + bx) * tmp_x + ax * zeta_x

    # Verlet time integration
    new_wfc = v_p ** 2 * inner_dt ** 2 * w_sum + 2.0 * wfc - wfp

    # Update PML auxiliary variables
    new_psi_y = by * dwfc_dy + ay * psi_y
    new_psi_x = bx * dwfc_dx + ax * psi_x
    new_zeta_y = by * tmp_y + ay * zeta_y
    new_zeta_x = bx * tmp_x + ax * zeta_x

    return new_wfc, new_psi_y, new_psi_x, new_zeta_y, new_zeta_x


# Compiled version of wave_step for GPU: fuses kernel calls and eliminates
# Python dispatch overhead (~4050 calls/shot → one CUDAGraph replay/step).
# Default mode (TorchInductor kernel fusion) is used instead of
# "reduce-overhead" (CUDAGraphs) because the FDTD loop feeds output tensors
# directly back as inputs, which conflicts with CUDAGraphs memory aliasing.
try:
    _wave_step_compiled = torch.compile(wave_step)
except Exception:
    _wave_step_compiled = wave_step


# ---------------------------------------------------------------------------
# FFT resampling (matches deepwave upsample / downsample)
# ---------------------------------------------------------------------------

def _fft_upsample(signal: torch.Tensor, step_ratio: int) -> torch.Tensor:
    """FFT-based low-pass upsampling of the last dimension by step_ratio.

    Matches deepwave.common.upsample (without time padding or tapering).
    """
    if step_ratio == 1:
        return signal
    nt = signal.shape[-1]
    up_nt = nt * step_ratio
    sig_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)
    # Zero Nyquist
    if sig_f.shape[-1] > 1:
        sig_f = sig_f.clone()
        sig_f[..., -1] = 0
    pad_len = up_nt // 2 + 1 - sig_f.shape[-1]
    if pad_len > 0:
        sig_f = F.pad(sig_f, (0, pad_len))
    return torch.fft.irfft(sig_f, n=up_nt, norm="ortho")


def _fft_downsample(signal: torch.Tensor, step_ratio: int) -> torch.Tensor:
    """FFT-based anti-aliased downsampling of the last dimension by step_ratio.

    Matches deepwave.common.downsample (without time padding or tapering).
    """
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
    in the padded model, following deepwave's convention.

    deepwave stores locations as [dim0_idx, dim1_idx] where dim0 is the slow
    (row) dimension and dim1 is the fast (column) dimension of the 2D model
    array, regardless of the physical interpretation of those axes.

    Args:
        loc: Shape (..., 2). Last dim is [dim0_idx, dim1_idx] where dim0 is
            the slow (row) dimension of the model array.
        pad_y: Padding added along the slow (row/dim0) dimension.
        pad_x: Padding added along the fast (col/dim1) dimension.
        nx_p: Width (number of columns) of the padded model.

    Returns:
        flat_idx: Shape (...), long tensor.
    """
    dim0_idx = loc[..., 0].long()   # slow / row dimension
    dim1_idx = loc[..., 1].long()   # fast / col dimension
    return ((dim0_idx + pad_y) * nx_p + (dim1_idx + pad_x)).long()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_acquisition_geometry(
    nx: int,
    n_shots: int = 10,
    n_receivers: int = 93,
    source_depth: int = 1,
    receiver_depth: int = 1,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """Create source and receiver location tensors for a surface acquisition.

    Args:
        nx: Number of horizontal grid points.
        n_shots: Number of shots. Default 10.
        n_receivers: Number of receivers per shot. Default 93.
        source_depth: Grid index in z for sources. Default 1.
        receiver_depth: Grid index in z for receivers. Default 1.
        device: PyTorch device.

    Returns:
        source_loc: Shape (n_shots, 1, 2) int64, last dim is [dim0_idx, dim1_idx]
            following deepwave's convention (dim0 = slow/row axis of model array).
        receiver_loc: Shape (n_shots, n_receivers, 2) int64.
    """
    source_loc = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_loc[..., 1] = source_depth
    source_loc[:, 0, 0] = torch.linspace(0, nx - 1, n_shots).round().long()

    receiver_loc = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    receiver_loc[..., 1] = receiver_depth
    receiver_loc[:, :, 0] = (
        torch.linspace(0, nx - 1, n_receivers).round().long()
        .unsqueeze(0)
        .expand(n_shots, -1)
    )
    return source_loc, receiver_loc


def make_ricker_wavelet(
    freq: float,
    nt: int,
    dt: float,
    n_shots: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate Ricker (Mexican-hat) wavelet source amplitudes for all shots.

    Formula: s(t) = (1 - 2π²f²(t-tp)²) * exp(-π²f²(t-tp)²)
    Peak time tp = 1.5 / freq.

    Args:
        freq: Central frequency in Hz.
        nt: Number of time samples (user-level, before step_ratio upsampling).
        dt: User-level time step in seconds.
        n_shots: Number of shots (wavelet replicated for each shot).
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
    spacing: tuple,
    dt: float,
    source_amp: torch.Tensor,
    source_loc: torch.Tensor,
    receiver_loc: torch.Tensor,
    freq: float,
    accuracy: int = 4,
    pml_width: int = 20,
    checkpoint_every: int = 64,
) -> torch.Tensor:
    """Simulate acoustic wave propagation and record receiver data.

    All shots are propagated in parallel: wavefields have shape
    (n_shots, ny_p, nx_p) so the GPU processes every shot simultaneously.
    wave_step broadcasts v_p (ny_p, nx_p) and the PML profiles over the
    shot batch dimension automatically.  Within the time loop, gradient
    checkpointing divides the nt_inner steps into segments of
    `checkpoint_every` steps, keeping peak GPU memory to roughly
    6 * n_shots * ny_p * nx_p * checkpoint_every * 4 bytes ≈ 2.5 GB for
    the full Marmousi survey (n_shots=10) on a 24 GB GPU.

    Args:
        v: Velocity model, shape (ny, nx), in m/s. May require grad.
        spacing: (dy, dx) grid spacing in meters.
        dt: User-level time step in seconds.
        source_amp: Source amplitudes, shape (n_shots, 1, nt_user).
        source_loc: Source positions, shape (n_shots, 1, 2), [x_idx, z_idx].
        receiver_loc: Receiver positions, shape (n_shots, n_rec, 2).
        freq: Dominant source frequency (Hz), used for PML tuning.
        accuracy: FD order (only 4 supported). Default 4.
        pml_width: PML width in grid cells on each side. Default 20.
        checkpoint_every: Steps per gradient-checkpoint segment. Default 64.
            Reduce for less memory; increase for faster computation.

    Returns:
        receiver_data: Recorded pressure at receivers, shape (n_shots, n_rec, nt_user).
    """
    assert accuracy == 4, "Only accuracy=4 (4th-order FD) is implemented."

    dy, dx = float(spacing[0]), float(spacing[1])
    ny, nx = v.shape
    fd_pad = accuracy // 2  # = 2

    # Padding = fd_pad + pml_width on each side
    pad = fd_pad + pml_width
    ny_p = ny + 2 * pad
    nx_p = nx + 2 * pad

    device = v.device
    dtype = v.dtype

    # CFL condition
    v_max = float(v.detach().abs().max().item())
    inner_dt, step_ratio = cfl_step_ratio(dy, dx, dt, v_max)

    # Pad velocity with replicate boundary (keeps v differentiable)
    v_p = F.pad(
        v.unsqueeze(0).unsqueeze(0),
        (pad, pad, pad, pad),
        mode="replicate",
    ).squeeze(0).squeeze(0)  # (ny_p, nx_p)

    # PML profiles (no gradient w.r.t. v)
    pml_profiles = setup_pml_profiles(
        ny_p, nx_p, pml_width, fd_pad,
        dy, dx, inner_dt, v_max,
        dtype, device, freq,
    )

    # Time parameters
    nt_user = source_amp.shape[2]
    nt_inner = nt_user * step_ratio

    # Upsample source: (n_shots, 1, nt_user) → (n_shots, 1, nt_inner)
    src_up = _fft_upsample(source_amp, step_ratio)  # (n_shots, 1, nt_inner)

    # Flat indices for sources and receivers in padded model
    src_flat = _loc_to_flat(source_loc, pad, pad, nx_p)   # (n_shots, 1)
    rec_flat = _loc_to_flat(receiver_loc, pad, pad, nx_p)  # (n_shots, n_rec)

    n_shots = source_amp.shape[0]
    flat_size = ny_p * nx_p

    # Use the compiled wave_step on CUDA to eliminate Python dispatch overhead.
    _step_fn = _wave_step_compiled if device.type == "cuda" else wave_step

    # ----------------------------------------------------------------- #
    # Run all shots in parallel (batched)                                #
    # Wavefields: (n_shots, ny_p, nx_p). wave_step broadcasts v_p       #
    # (ny_p, nx_p) and pml_profiles over the shot batch dimension.      #
    # ----------------------------------------------------------------- #

    # Scale source: -v(xs)^2 * inner_dt^2, differentiable through v_p.
    # v_xs: (n_shots,) — velocity at each shot's source grid point.
    v_xs = v_p.reshape(-1)[src_flat[:, 0]]                              # (n_shots,)
    src_scaled = -src_up[:, 0, :] * (v_xs ** 2).unsqueeze(1) * (inner_dt ** 2)
    # shape: (n_shots, nt_inner)

    # Source spatial indicator: 1 at each shot's source location, 0 elsewhere.
    src_indicator = torch.zeros(n_shots, flat_size, device=device, dtype=dtype)
    src_indicator.scatter_(1, src_flat, 1.0)                            # (n_shots, flat_size)
    src_indicator = src_indicator.view(n_shots, ny_p, nx_p)             # (n_shots, ny_p, nx_p)

    # Row index for batched receiver extraction: wfc[shot, rec_flat[shot, :]]
    shot_idx = torch.arange(n_shots, device=device)                     # (n_shots,)

    # Initialize batched wavefields: (n_shots, ny_p, nx_p)
    wfc   = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    wfp   = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    psi_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_y = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)
    zeta_x = torch.zeros(n_shots, ny_p, nx_p, device=device, dtype=dtype)

    def _make_segment(t_start: int, k: int):
        """Return a closure that runs k batched steps starting at t_start."""
        def _segment(wfc, wfp, psi_y, psi_x, zeta_y, zeta_x):
            rec_list: List[torch.Tensor] = []
            for i in range(k):
                t = t_start + i
                # Extract receivers for all shots: (n_shots, n_rec)
                rec_list.append(
                    wfc.reshape(n_shots, -1)[shot_idx.unsqueeze(1), rec_flat]
                )
                wfc_new, psi_y, psi_x, zeta_y, zeta_x = _step_fn(
                    v_p, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                    pml_profiles, dy, dx, inner_dt,
                )
                # Inject source: (n_shots,1,1) * (n_shots,ny_p,nx_p)
                wfc_new = wfc_new + src_scaled[:, t].reshape(n_shots, 1, 1) * src_indicator
                wfp = wfc
                wfc = wfc_new
            # Stack: (k, n_shots, n_rec)
            rec = torch.stack(rec_list, dim=0)
            return wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec
        return _segment

    rec_segments: List[torch.Tensor] = []

    for seg_start in range(0, nt_inner, checkpoint_every):
        k = min(checkpoint_every, nt_inner - seg_start)
        seg_fn = _make_segment(seg_start, k)

        if v.requires_grad:
            # Use gradient checkpointing: don't store activations during forward
            wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec = _checkpoint(
                seg_fn, wfc, wfp, psi_y, psi_x, zeta_y, zeta_x,
                use_reentrant=False,
            )
        else:
            # No gradient needed: run directly (faster, no checkpointing overhead)
            with torch.no_grad():
                wfc, wfp, psi_y, psi_x, zeta_y, zeta_x, rec = seg_fn(
                    wfc, wfp, psi_y, psi_x, zeta_y, zeta_x
                )

        rec_segments.append(rec)

    # Assemble: (nt_inner, n_shots, n_rec) → (n_shots, n_rec, nt_inner)
    rec_tensor = torch.cat(rec_segments, dim=0).permute(1, 2, 0).contiguous()
    # FFT downsample: (n_shots, n_rec, nt_inner) → (n_shots, n_rec, nt_user)
    return _fft_downsample(rec_tensor, step_ratio)
