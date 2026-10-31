"""
Wave-Optics Forward Model for the Light-Field Microscope Task.

Compact Python port of the core LFM forward model from oLaF's MATLAB
implementation, adapted for the USAF benchmark. Includes dataclasses,
the LFMSystem forward/backward projector, all wave-optics physics functions,
and high-level wrappers used by the reconstruction pipeline.

Source MATLAB files used as reference:
- Code/internal/util/CameraGeometry/LFM_setCameraParams.m
- Code/internal/util/CameraGeometry/LFM_computeResolution.m
- Code/internal/LFPSF/LFM_computePSFsize.m
- Code/internal/LFPSF/LFM_calcPSF.m
- Code/internal/LFPSF/LFM_ulensTransmittance.m
- Code/internal/LFPSF/LFM_mlaTransmittance.m
- Code/internal/projectionOperators/LFM_forwardProject.m

Deliberate simplifications for a single-file Python implementation:
- Fixed to the regular-grid, single-focus, plenoptic-1 USAF setup.
- Lenslet centers are generated analytically from a regular grid instead of
  being calibrated from a white image.
- The backward projector is implemented as the adjoint of the forward model
  instead of explicitly precomputing Ht via backward-project-single-point.
- The full kernel bank is materialized for readability instead of MATLAB's
  quarter-symmetry optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import scipy.fft as fft
from scipy import signal
from scipy.special import j0
import yaml


EPS = 1e-8


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    grid_type: str
    focus: str
    plenoptic: int
    u_lens_mask: int
    magnification: float
    numerical_aperture: float
    tube_focal_length: float
    microlens_focal_length: float
    tube_to_mla: float
    mla_to_sensor: float
    lens_pitch: float
    pixel_pitch: float
    wavelength: float
    refractive_index: float
    range_mode: str
    fobj: float
    delta_ot: float
    spacing_px: float
    new_spacing_px: int
    new_pixel_pitch: float
    wave_number: float
    objective_radius: float
    ulens_radius: float
    tube_radius: float
    dof: float
    offset_fobj: float
    mla_magnification: float


@dataclass
class Resolution:
    nnum: Tuple[int, int]
    nnum_half: Tuple[int, int]
    tex_nnum: Tuple[int, int]
    tex_nnum_half: Tuple[int, int]
    sensor_res: Tuple[float, float]
    tex_res: Tuple[float, float, float]
    tex_scale_factor: Tuple[float, float]
    sens_mask: np.ndarray
    tex_mask: np.ndarray
    depths: np.ndarray
    depth_step: float
    depth_range: Tuple[float, float]
    yspace: np.ndarray
    xspace: np.ndarray
    y_ml_space: np.ndarray
    x_ml_space: np.ndarray
    psf_size_lenslets: int


@dataclass
class PatternOp:
    local_coord: Tuple[int, int]
    tex_indices: np.ndarray
    img_indices: np.ndarray
    kernels: List[np.ndarray]
    adjoint_kernels: List[np.ndarray]


# ---------------------------------------------------------------------------
# LFMSystem: forward and backward projectors
# ---------------------------------------------------------------------------

class LFMSystem:
    def __init__(
        self,
        camera: CameraParams,
        resolution: Resolution,
        lenslet_rows_px: np.ndarray,
        lenslet_cols_px: np.ndarray,
        img_shape: Tuple[int, int],
        tex_shape: Tuple[int, int],
        pattern_ops: Sequence[PatternOp],
    ) -> None:
        self.camera = camera
        self.resolution = resolution
        self.lenslet_rows_px = lenslet_rows_px
        self.lenslet_cols_px = lenslet_cols_px
        self.img_shape = img_shape
        self.tex_shape = tex_shape
        self.pattern_ops = list(pattern_ops)
        self.depths = resolution.depths

    def forward_project(self, volume: np.ndarray) -> np.ndarray:
        projection = np.zeros(self.img_shape, dtype=np.float64)
        volume_slices = [np.asarray(volume[:, :, d], dtype=np.float64) for d in range(volume.shape[2])]

        for depth_idx, plane in enumerate(volume_slices):
            plane_flat = plane.ravel()
            for op in self.pattern_ops:
                sampled = plane_flat[op.tex_indices]
                if not np.any(sampled):
                    continue
                sparse_plane = np.zeros(self.img_shape[0] * self.img_shape[1], dtype=np.float64)
                sparse_plane[op.img_indices] = sampled
                sparse_plane = sparse_plane.reshape(self.img_shape)
                projection += signal.fftconvolve(sparse_plane, op.kernels[depth_idx], mode="same")

        projection[projection < 0] = 0
        return projection

    def backward_project(self, projection: np.ndarray) -> np.ndarray:
        backprojection = np.zeros(self.tex_shape + (len(self.depths),), dtype=np.float64)
        projection = np.asarray(projection, dtype=np.float64)

        for depth_idx in range(len(self.depths)):
            back_flat = np.zeros(self.tex_shape[0] * self.tex_shape[1], dtype=np.float64)
            for op in self.pattern_ops:
                conv_result = signal.fftconvolve(projection, op.adjoint_kernels[depth_idx], mode="same")
                back_flat[op.tex_indices] += conv_result.ravel()[op.img_indices]
            backprojection[:, :, depth_idx] = back_flat.reshape(self.tex_shape)

        backprojection[backprojection < 0] = 0
        return backprojection


# ---------------------------------------------------------------------------
# Camera parameter loading
# ---------------------------------------------------------------------------

def load_camera_params(config_path: Path, new_spacing_px: int) -> CameraParams:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    fm_value = raw["fm"]
    if isinstance(fm_value, (list, tuple)):
        if len(fm_value) != 1:
            raise ValueError("This compact Python port only supports single-focus microlenses.")
        fm_value = fm_value[0]

    grid_type = str(raw["gridType"])
    focus = str(raw["focus"])
    plenoptic = int(raw["plenoptic"])
    if grid_type != "reg" or focus != "single" or plenoptic != 1:
        raise ValueError("This script is fixed to regular-grid, single-focus, plenoptic-1 USAF LFM.")

    magnification = float(raw["M"])
    numerical_aperture = float(raw["NA"])
    tube_focal_length = float(raw["ftl"])
    microlens_focal_length = float(fm_value)
    tube_to_mla = float(raw["tube2mla"]) if float(raw["tube2mla"]) != 0 else tube_focal_length
    mla_to_sensor = float(raw["mla2sensor"]) if float(raw["mla2sensor"]) != 0 else microlens_focal_length
    lens_pitch = float(raw["lensPitch"])
    pixel_pitch = float(raw["pixelPitch"])
    wavelength = float(raw["WaveLength"])
    refractive_index = float(raw["n"])

    spacing_px = lens_pitch / pixel_pitch
    fobj = tube_focal_length / magnification
    delta_ot = tube_focal_length + fobj
    wave_number = 2 * math.pi * refractive_index / wavelength
    objective_radius = fobj * numerical_aperture
    tube_radius = objective_radius
    dof = fobj
    offset_fobj = dof - fobj
    ulens_radius = tube_radius * mla_to_sensor / tube_to_mla
    range_mode = "quarter" if grid_type == "reg" else "full"

    return CameraParams(
        grid_type=grid_type,
        focus=focus,
        plenoptic=plenoptic,
        u_lens_mask=int(raw["uLensMask"]),
        magnification=magnification,
        numerical_aperture=numerical_aperture,
        tube_focal_length=tube_focal_length,
        microlens_focal_length=microlens_focal_length,
        tube_to_mla=tube_to_mla,
        mla_to_sensor=mla_to_sensor,
        lens_pitch=lens_pitch,
        pixel_pitch=pixel_pitch,
        wavelength=wavelength,
        refractive_index=refractive_index,
        range_mode=range_mode,
        fobj=fobj,
        delta_ot=delta_ot,
        spacing_px=spacing_px,
        new_spacing_px=int(new_spacing_px),
        new_pixel_pitch=lens_pitch / float(new_spacing_px),
        wave_number=wave_number,
        objective_radius=objective_radius,
        ulens_radius=ulens_radius,
        tube_radius=tube_radius,
        dof=dof,
        offset_fobj=offset_fobj,
        mla_magnification=magnification,
    )


# ---------------------------------------------------------------------------
# Optics helpers
# ---------------------------------------------------------------------------

def fix_mask(mask: np.ndarray, spacing: Tuple[int, int], grid_type: str) -> np.ndarray:
    mask = mask.astype(np.uint8).copy()
    trial_space = np.zeros((mask.shape[0] * 3, mask.shape[1] * 3), dtype=np.float64)
    r_center = trial_space.shape[0] // 2
    c_center = trial_space.shape[1] // 2
    rs, cs = spacing

    if grid_type != "reg":
        raise ValueError("This script only supports regular grids.")

    row_positions = [r_center - rs, r_center, r_center + rs]
    col_positions = [c_center - cs, c_center, c_center + cs]
    trial_space[np.ix_(row_positions, col_positions)] = 1

    space = signal.convolve2d(trial_space, mask.astype(np.float64), mode="same")
    r, c = mask.shape
    space_center = space[r : 2 * r, c : 2 * c]

    new_mask = mask.copy()
    for i in range(r):
        for j in range(c):
            if space_center[i, j] == 0:
                new_mask[i, j] = 1
                space = signal.convolve2d(trial_space, new_mask.astype(np.float64), mode="same")
                space_center = space[r : 2 * r, c : 2 * c]
            if space_center[i, j] == 2:
                new_mask[i, j] = 0
                space = signal.convolve2d(trial_space, new_mask.astype(np.float64), mode="same")
                space_center = space[r : 2 * r, c : 2 * c]

    return new_mask.astype(bool)


def compute_patch_mask(
    spacing: Tuple[int, int],
    grid_type: str,
    res: Tuple[float, float],
    patch_radius: float,
    nnum: Tuple[int, int],
) -> np.ndarray:
    ys = np.arange(-math.floor(nnum[0] / 2), math.floor(nnum[0] / 2) + 1, dtype=np.float64)
    xs = np.arange(-math.floor(nnum[1] / 2), math.floor(nnum[1] / 2) + 1, dtype=np.float64)
    yy, xx = np.meshgrid(res[0] * ys, res[1] * xs, indexing="ij")
    mask = np.sqrt(yy * yy + xx * xx) < patch_radius
    return fix_mask(mask, spacing, grid_type)


def compute_psf_size_lenslets(max_depth: float, camera: CameraParams) -> int:
    max_depth = max_depth - camera.offset_fobj
    zobj = camera.fobj - max_depth
    if math.isclose(zobj, camera.fobj) or math.isclose(zobj, camera.dof):
        zobj += 1e-5 * camera.fobj

    z1 = (zobj * camera.fobj) / (zobj - camera.fobj)
    tube_radius = camera.objective_radius * camera.delta_ot * abs((1 / z1) - (1 / camera.delta_ot))
    z2 = camera.tube_focal_length * (camera.delta_ot - z1) / (camera.delta_ot - z1 - camera.tube_focal_length)
    blur_radius = tube_radius * camera.tube_to_mla * abs((1 / z2) - (1 / camera.tube_to_mla))
    return int(math.ceil(blur_radius / camera.lens_pitch) + 2)


def build_resolution(
    camera: CameraParams,
    depth_range: Tuple[float, float],
    depth_step: float,
) -> Resolution:
    nspacing_lenslet = (camera.new_spacing_px, camera.new_spacing_px)
    sensor_res = (
        camera.lens_pitch / nspacing_lenslet[0],
        camera.lens_pitch / nspacing_lenslet[1],
    )
    nnum = (
        nspacing_lenslet[0] + (1 - nspacing_lenslet[0] % 2),
        nspacing_lenslet[1] + (1 - nspacing_lenslet[1] % 2),
    )
    tex_nnum = nnum
    tex_scale_factor = (1.0, 1.0)
    tex_res = (
        sensor_res[0] / (tex_scale_factor[0] * camera.magnification),
        sensor_res[1] / (tex_scale_factor[1] * camera.magnification),
        depth_step,
    )
    sens_mask = compute_patch_mask(nspacing_lenslet, camera.grid_type, sensor_res, camera.ulens_radius, nnum)
    tex_mask = compute_patch_mask(
        nspacing_lenslet,
        camera.grid_type,
        (tex_res[0], tex_res[1]),
        tex_nnum[0] * tex_res[0] / 2.0,
        tex_nnum,
    )

    depths = np.arange(depth_range[0], depth_range[1] + 0.5 * depth_step, depth_step, dtype=np.float64)
    max_depth = float(depths[np.argmax(np.abs(depths + camera.offset_fobj))] + camera.offset_fobj)
    psf_size_lenslets = compute_psf_size_lenslets(max_depth, camera)
    img_half = max(nnum[1] * psf_size_lenslets, 2 * nnum[1])
    yspace = sensor_res[0] * np.arange(-img_half, img_half + 1, dtype=np.float64)
    xspace = sensor_res[1] * np.arange(-img_half, img_half + 1, dtype=np.float64)
    y_ml_space = sensor_res[0] * np.arange(-(nnum[0] // 2), (nnum[0] // 2) + 1, dtype=np.float64)
    x_ml_space = sensor_res[1] * np.arange(-(nnum[1] // 2), (nnum[1] // 2) + 1, dtype=np.float64)

    return Resolution(
        nnum=nnum,
        nnum_half=((nnum[0] + 1) // 2, (nnum[1] + 1) // 2),
        tex_nnum=tex_nnum,
        tex_nnum_half=((tex_nnum[0] + 1) // 2, (tex_nnum[1] + 1) // 2),
        sensor_res=sensor_res,
        tex_res=tex_res,
        tex_scale_factor=tex_scale_factor,
        sens_mask=sens_mask,
        tex_mask=tex_mask,
        depths=depths,
        depth_step=depth_step,
        depth_range=depth_range,
        yspace=yspace,
        xspace=xspace,
        y_ml_space=y_ml_space,
        x_ml_space=x_ml_space,
        psf_size_lenslets=psf_size_lenslets,
    )


def build_regular_lenslet_grid(n_lenslets: int, spacing_px: int) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.arange(n_lenslets, dtype=np.int32) * int(spacing_px)
    coords = coords - coords[n_lenslets // 2]
    rows, cols = np.meshgrid(coords, coords, indexing="ij")
    return rows, cols


def select_used_centers(
    lenslet_rows: np.ndarray,
    lenslet_cols: np.ndarray,
    psf_size_lenslets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    used_radius = psf_size_lenslets + 3
    center_row = lenslet_rows.shape[0] // 2
    center_col = lenslet_rows.shape[1] // 2
    row_lo = max(center_row - used_radius, 0)
    row_hi = min(center_row + used_radius + 1, lenslet_rows.shape[0])
    col_lo = max(center_col - used_radius, 0)
    col_hi = min(center_col + used_radius + 1, lenslet_rows.shape[1])
    return lenslet_rows[row_lo:row_hi, col_lo:col_hi], lenslet_cols[row_lo:row_hi, col_lo:col_hi]


def integer_shift(image: np.ndarray, shift_row: int, shift_col: int) -> np.ndarray:
    shift_row = int(round(shift_row))
    shift_col = int(round(shift_col))
    out = np.zeros_like(image)

    src_row_start = max(0, -shift_row)
    src_row_end = image.shape[0] - max(0, shift_row)
    dst_row_start = max(0, shift_row)
    dst_row_end = dst_row_start + (src_row_end - src_row_start)

    src_col_start = max(0, -shift_col)
    src_col_end = image.shape[1] - max(0, shift_col)
    dst_col_start = max(0, shift_col)
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    if src_row_end <= src_row_start or src_col_end <= src_col_start:
        return out

    out[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = image[
        src_row_start:src_row_end,
        src_col_start:src_col_end,
    ]
    return out


def compute_native_plane_psf(
    depth_um: float,
    camera: CameraParams,
    resolution: Resolution,
    theta_samples: int,
) -> np.ndarray:
    alpha = math.asin(camera.numerical_aperture / camera.refractive_index)
    sin_alpha_half_sq = math.sin(alpha / 2.0) ** 2
    demag = 1.0 / camera.magnification
    d1 = camera.dof - depth_um
    k = camera.wave_number
    u = 4.0 * k * depth_um * sin_alpha_half_sq
    koi = demag / ((d1 * camera.wavelength) ** 2) * np.exp(-1j * u / (4.0 * sin_alpha_half_sq))

    yy, xx = np.meshgrid(resolution.yspace, resolution.xspace, indexing="ij")
    radial = np.sqrt(yy * yy + xx * xx) / camera.magnification
    v = k * radial * math.sin(alpha)

    theta = np.linspace(0.0, alpha, theta_samples, dtype=np.float64)
    theta = theta[:, None, None]
    sin_theta_half_sq = np.sin(theta / 2.0) ** 2
    phase = np.exp(1j * (u / 2.0) * sin_theta_half_sq / sin_alpha_half_sq)
    prefactor = np.sqrt(np.cos(theta)) * (1.0 + np.cos(theta)) * np.sin(theta)
    bessel_arg = (np.sin(theta) / math.sin(alpha)) * v[None, :, :]
    integrand = prefactor * phase * j0(bessel_arg)
    integral = np.trapz(integrand, x=theta[:, 0, 0], axis=0)
    return koi * integral


def prop_to_sensor(
    field_at_mla: np.ndarray,
    sensor_res: Tuple[float, float],
    z_um: float,
    wavelength_um: float,
) -> np.ndarray:
    ny, nx = field_at_mla.shape
    k = 2 * math.pi / wavelength_um
    fy = fft.fftfreq(ny, d=sensor_res[0])
    fx = fft.fftfreq(nx, d=sensor_res[1])
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    argument = (1.0 - (wavelength_um ** 2) * (fy_grid ** 2 + fx_grid ** 2)).astype(np.complex128)
    transfer = np.exp(1j * np.sqrt(argument) * z_um * k)
    return np.exp(1j * k * z_um) * fft.ifft2(fft.fft2(field_at_mla) * transfer)


def build_ulens_pattern(camera: CameraParams, resolution: Resolution) -> np.ndarray:
    yy, xx = np.meshgrid(resolution.y_ml_space, resolution.x_ml_space, indexing="ij")
    phase = np.exp(-1j * camera.wave_number / (2.0 * camera.microlens_focal_length) * (yy * yy + xx * xx))
    if camera.u_lens_mask == 1:
        phase = np.where(resolution.sens_mask, phase, 0.0)
    else:
        radius = np.sqrt(yy * yy + xx * xx)
        phase = np.where(radius < (camera.lens_pitch / 2.0 - 3.0), phase, 0.0)
    return phase


def overlay_centered_patch(canvas: np.ndarray, patch: np.ndarray, center_row: int, center_col: int) -> None:
    half_rows = patch.shape[0] // 2
    half_cols = patch.shape[1] // 2

    row_lo = center_row - half_rows
    row_hi = row_lo + patch.shape[0]
    col_lo = center_col - half_cols
    col_hi = col_lo + patch.shape[1]

    clip_row_lo = max(row_lo, 0)
    clip_row_hi = min(row_hi, canvas.shape[0])
    clip_col_lo = max(col_lo, 0)
    clip_col_hi = min(col_hi, canvas.shape[1])

    if clip_row_lo >= clip_row_hi or clip_col_lo >= clip_col_hi:
        return

    patch_row_lo = clip_row_lo - row_lo
    patch_row_hi = patch_row_lo + (clip_row_hi - clip_row_lo)
    patch_col_lo = clip_col_lo - col_lo
    patch_col_hi = patch_col_lo + (clip_col_hi - clip_col_lo)

    canvas[clip_row_lo:clip_row_hi, clip_col_lo:clip_col_hi] += patch[
        patch_row_lo:patch_row_hi,
        patch_col_lo:patch_col_hi,
    ]


def build_mla_array(
    used_rows_px: np.ndarray,
    used_cols_px: np.ndarray,
    resolution: Resolution,
    ulens_pattern: np.ndarray,
) -> np.ndarray:
    y_length = len(resolution.yspace)
    x_length = len(resolution.xspace)
    y_extended = y_length + 2 * len(resolution.y_ml_space)
    x_extended = x_length + 2 * len(resolution.x_ml_space)
    canvas = np.zeros((y_extended, x_extended), dtype=np.complex128)
    center_row = y_extended // 2
    center_col = x_extended // 2

    for row_offset, col_offset in zip(used_rows_px.ravel(), used_cols_px.ravel()):
        overlay_centered_patch(canvas, ulens_pattern, int(round(center_row + row_offset)), int(round(center_col + col_offset)))

    row_lo = (y_extended // 2) - (y_length // 2)
    row_hi = row_lo + y_length
    col_lo = (x_extended // 2) - (x_length // 2)
    col_hi = col_lo + x_length
    return canvas[row_lo:row_hi, col_lo:col_hi]


def build_forward_kernels(
    camera: CameraParams,
    resolution: Resolution,
    used_rows_px: np.ndarray,
    used_cols_px: np.ndarray,
    theta_samples: int,
    clamp_tol: float,
) -> List[List[np.ndarray]]:
    ulens_pattern = build_ulens_pattern(camera, resolution)
    mla_array = build_mla_array(used_rows_px, used_cols_px, resolution, ulens_pattern)

    psf_stack = []
    for depth_um in resolution.depths + camera.offset_fobj:
        psf_stack.append(compute_native_plane_psf(float(depth_um), camera, resolution, theta_samples))
    print(f"[PSF] built native PSF stack for depths {resolution.depths.tolist()}")

    kernels: List[List[np.ndarray]] = []
    row_center = resolution.tex_nnum[0] // 2
    col_center = resolution.tex_nnum[1] // 2
    for local_row in range(resolution.tex_nnum[0]):
        row_kernels: List[np.ndarray] = []
        for local_col in range(resolution.tex_nnum[1]):
            shift_row = local_row - row_center
            shift_col = local_col - col_center
            depth_kernels: List[np.ndarray] = []
            for psf_ref in psf_stack:
                psf_shift = integer_shift(psf_ref, shift_row, shift_col)
                psf_mla = psf_shift * mla_array
                sensor_field = prop_to_sensor(psf_mla, resolution.sensor_res, camera.mla_to_sensor, camera.wavelength)
                sensor_field = integer_shift(sensor_field, -shift_row, -shift_col)
                kernel = np.abs(sensor_field) ** 2
                if kernel.max() > 0:
                    kernel[kernel < kernel.max() * clamp_tol] = 0
                    kernel_sum = kernel.sum()
                    if kernel_sum > 0:
                        kernel = kernel / kernel_sum
                depth_kernels.append(kernel.astype(np.float64))
            row_kernels.append(depth_kernels)
        kernels.append(row_kernels)
        print(f"[Kernel] row {local_row + 1:02d}/{resolution.tex_nnum[0]}")

    return kernels


def build_pattern_ops(
    kernels: Sequence[Sequence[Sequence[np.ndarray]]],
    lenslet_rows_px: np.ndarray,
    lenslet_cols_px: np.ndarray,
    img_shape: Tuple[int, int],
    tex_shape: Tuple[int, int],
    resolution: Resolution,
) -> List[PatternOp]:
    pattern_ops: List[PatternOp] = []
    img_offset_row = img_shape[0] // 2
    img_offset_col = img_shape[1] // 2
    tex_offset_row = tex_shape[0] // 2
    tex_offset_col = tex_shape[1] // 2
    local_row_center = resolution.tex_nnum[0] // 2
    local_col_center = resolution.tex_nnum[1] // 2

    for local_row in range(resolution.tex_nnum[0]):
        for local_col in range(resolution.tex_nnum[1]):
            if not resolution.tex_mask[local_row, local_col]:
                continue

            row_indices = lenslet_rows_px + tex_offset_row + (local_row - local_row_center)
            col_indices = lenslet_cols_px + tex_offset_col + (local_col - local_col_center)
            img_rows = lenslet_rows_px + img_offset_row + (local_row - local_row_center)
            img_cols = lenslet_cols_px + img_offset_col + (local_col - local_col_center)

            valid = (
                (row_indices >= 0)
                & (row_indices < tex_shape[0])
                & (col_indices >= 0)
                & (col_indices < tex_shape[1])
                & (img_rows >= 0)
                & (img_rows < img_shape[0])
                & (img_cols >= 0)
                & (img_cols < img_shape[1])
            )
            tex_indices = np.ravel_multi_index(
                (row_indices[valid].astype(np.int64), col_indices[valid].astype(np.int64)),
                tex_shape,
            )
            img_indices = np.ravel_multi_index(
                (img_rows[valid].astype(np.int64), img_cols[valid].astype(np.int64)),
                img_shape,
            )
            depth_kernels = [kernels[local_row][local_col][depth_idx] for depth_idx in range(len(resolution.depths))]
            adjoint_kernels = [kernel[::-1, ::-1].copy() for kernel in depth_kernels]
            pattern_ops.append(
                PatternOp(
                    local_coord=(local_row, local_col),
                    tex_indices=tex_indices,
                    img_indices=img_indices,
                    kernels=depth_kernels,
                    adjoint_kernels=adjoint_kernels,
                )
            )

    print(f"[Pattern] active local coordinates: {len(pattern_ops)} / {resolution.tex_nnum[0] * resolution.tex_nnum[1]}")
    return pattern_ops


def build_lfm_system(
    config_path: Path,
    n_lenslets: int,
    new_spacing_px: int,
    depth_range: Tuple[float, float],
    depth_step: float,
    theta_samples: int,
    clamp_tol: float,
) -> LFMSystem:
    camera = load_camera_params(config_path, new_spacing_px=new_spacing_px)
    resolution = build_resolution(camera, depth_range=depth_range, depth_step=depth_step)
    lenslet_rows_px, lenslet_cols_px = build_regular_lenslet_grid(n_lenslets=n_lenslets, spacing_px=new_spacing_px)
    used_rows_px, used_cols_px = select_used_centers(
        lenslet_rows_px,
        lenslet_cols_px,
        psf_size_lenslets=resolution.psf_size_lenslets,
    )

    img_shape = (n_lenslets * new_spacing_px, n_lenslets * new_spacing_px)
    tex_shape = img_shape

    kernels = build_forward_kernels(
        camera=camera,
        resolution=resolution,
        used_rows_px=used_rows_px,
        used_cols_px=used_cols_px,
        theta_samples=theta_samples,
        clamp_tol=clamp_tol,
    )
    pattern_ops = build_pattern_ops(
        kernels=kernels,
        lenslet_rows_px=lenslet_rows_px,
        lenslet_cols_px=lenslet_cols_px,
        img_shape=img_shape,
        tex_shape=tex_shape,
        resolution=resolution,
    )
    print(
        "[System] "
        f"img_shape={img_shape}, tex_shape={tex_shape}, "
        f"sensor_res={resolution.sensor_res}, tex_res={resolution.tex_res}, "
        f"depths={resolution.depths.tolist()}"
    )

    return LFMSystem(
        camera=camera,
        resolution=resolution,
        lenslet_rows_px=lenslet_rows_px,
        lenslet_cols_px=lenslet_cols_px,
        img_shape=img_shape,
        tex_shape=tex_shape,
        pattern_ops=pattern_ops,
    )


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

def write_wave_model_config(metadata: dict, config_path: str | Path) -> None:
    """Write the YAML config consumed by load_camera_params."""
    microscope = metadata["microscope"]
    mla = metadata["mla"]
    payload = {
        "gridType": str(mla["gridType"]),
        "focus": str(mla["focus"]),
        "plenoptic": int(mla["plenoptic"]),
        "uLensMask": int(mla["uLensMask"]),
        "M": float(microscope["M"]),
        "NA": float(microscope["NA"]),
        "ftl": float(microscope["ftl"]),
        "fm": float(mla["fm"]),
        "tube2mla": float(mla.get("tube2mla", 0.0)),
        "mla2sensor": float(mla.get("mla2sensor", 0.0)),
        "lensPitch": float(mla["lensPitch"]),
        "pixelPitch": float(mla["pixelPitch"]),
        "WaveLength": float(microscope["WaveLength"]),
        "n": float(microscope["n"]),
    }
    path = Path(config_path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_volume_system(
    config_path: str | Path,
    n_lenslets: int,
    new_spacing_px: int,
    depth_range_um: tuple[float, float],
    depth_step_um: float,
    theta_samples: int,
    kernel_tol: float,
) -> LFMSystem:
    """Construct the wave-model light-field system for the requested depth grid."""
    return build_lfm_system(
        config_path=Path(config_path),
        n_lenslets=int(n_lenslets),
        new_spacing_px=int(new_spacing_px),
        depth_range=(float(depth_range_um[0]), float(depth_range_um[1])),
        depth_step=float(depth_step_um),
        theta_samples=int(theta_samples),
        clamp_tol=float(kernel_tol),
    )


def compute_conventional_image(
    system,
    object_2d: np.ndarray,
    target_depth_um: float,
    theta_samples: int,
) -> np.ndarray:
    """Compute the defocused conventional-microscope baseline image."""
    psf_wave = compute_native_plane_psf(
        float(target_depth_um + system.camera.offset_fobj),
        system.camera,
        system.resolution,
        int(theta_samples),
    )
    psf_intensity = np.abs(psf_wave) ** 2
    psf_sum = float(psf_intensity.sum())
    if psf_sum > 0:
        psf_intensity /= psf_sum
    return signal.fftconvolve(
        np.asarray(object_2d, dtype=np.float64),
        np.asarray(psf_intensity, dtype=np.float64),
        mode="same",
    ).astype(np.float64)
