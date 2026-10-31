"""Generate synthetic Shack-Hartmann wavefront sensing data.

Simulates a VLT-like 8 m telescope with a Shack-Hartmann WFS and a
disk-harmonic DM.  For each of N_WFE_LEVELS target RMS wavefront errors,
a random DM mode combination is generated, applied to the WFS (with Poisson
noise), and both the raw detector image and the ground-truth wavefront are saved.

The raw SH-WFS detector images (not pre-extracted slopes) are saved so that
centroid estimation forms part of the reconstruction pipeline.

Produces
--------
data/raw_data.npz
    response_matrix    (1, N_slopes, N_modes)        float32
    wfs_images         (1, N_levels, H_det, W_det)   float32   [H_det=W_det=128]
    ref_image          (1, H_det, W_det)              float32
    detector_coords_x  (1, H_det, W_det)              float32
    detector_coords_y  (1, H_det, W_det)              float32
    subap_map          (1, H_det, W_det)              int32
    dm_modes           (1, N_modes, N_pupil_px)       float32
    aperture           (1, N_pupil_px)                float32

data/ground_truth.npz
    wavefront_phases   (1, N_levels, N_pupil_px)  float32  [rad at lambda_wfs]

data/meta_data.json
    Optical and instrument parameters; wfe_levels_nm list.

Usage
-----
    cd tasks/shack-hartmann
    python src/generate_data.py

Dependencies
------------
This script requires HCIPy for the optical simulation.  HCIPy is NOT listed in
requirements.txt because it is only needed to regenerate the data files; the
reconstruction pipeline (main.py, src/) runs on numpy/scipy alone.

To regenerate the data:
    pip install hcipy>=0.6.0
    python src/generate_data.py
"""

import os
import json
import numpy as np
import scipy.ndimage as ndimage

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

from hcipy import (
    make_pupil_grid, make_focal_grid,
    make_obstructed_circular_aperture, evaluate_supersampled,
    Wavefront,
    SquareShackHartmannWavefrontSensorOptics,
    ShackHartmannWavefrontSensorEstimator,
    NoiselessDetector,
    Magnifier,
    make_disk_harmonic_basis,
    ModeBasis,
    DeformableMirror,
    large_poisson,
)

# ── Instrument parameters ─────────────────────────────────────────────────────
TELESCOPE_DIAMETER = 8.0        # m
CENTRAL_OBS        = 1.2        # m
CENTRAL_OBS_RATIO  = CENTRAL_OBS / TELESCOPE_DIAMETER
SPIDER_WIDTH       = 0.05       # m
OVERSIZING         = 16 / 15

NUM_PUPIL_PIXELS   = 128
PUPIL_GRID_DIAM    = TELESCOPE_DIAMETER * OVERSIZING

WAVELENGTH_WFS     = 0.7e-6     # m  (R-band)

F_NUMBER           = 50
NUM_LENSLETS       = 20
SH_DIAMETER        = 5e-3       # m
MAGNIFICATION      = SH_DIAMETER / TELESCOPE_DIAMETER

NUM_MODES          = 150        # DM modes

# Star flux
ZERO_MAG_FLUX      = 3.9e10     # photons/s
STELLAR_MAGNITUDE  = 5

# Calibration
PROBE_AMP_WL_FRAC  = 0.01

# ── Wavefront error levels ────────────────────────────────────────────────────
# Single-pass OPD RMS in nm: WFE_nm = sigma(phi) * lambda / (2*pi)
WFE_LEVELS_NM = [50, 100, 200, 400]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    np.random.seed(42)

    print("=== Generating Shack-Hartmann wavefront sensing data ===")

    # ------------------------------------------------------------------
    # 1. Pupil grid and aperture
    # ------------------------------------------------------------------
    pupil_grid  = make_pupil_grid(NUM_PUPIL_PIXELS, PUPIL_GRID_DIAM)
    aper_gen    = make_obstructed_circular_aperture(
        TELESCOPE_DIAMETER, CENTRAL_OBS_RATIO,
        num_spiders=4, spider_width=SPIDER_WIDTH)
    aperture    = evaluate_supersampled(aper_gen, pupil_grid, 4)
    N_pupil_px  = pupil_grid.size
    aperture_np = np.array(aperture, dtype=np.float32)
    print(f"Pupil grid: {NUM_PUPIL_PIXELS}×{NUM_PUPIL_PIXELS} = {N_pupil_px} px")

    flux_per_frame = ZERO_MAG_FLUX * 10 ** (-STELLAR_MAGNITUDE / 2.5)
    print(f"WFS photons/frame: {flux_per_frame:.0f}")

    # ------------------------------------------------------------------
    # 2. SH-WFS optics and focal grid
    # ------------------------------------------------------------------
    magnifier = Magnifier(MAGNIFICATION)
    shwfs     = SquareShackHartmannWavefrontSensorOptics(
        pupil_grid.scaled(MAGNIFICATION), F_NUMBER, NUM_LENSLETS, SH_DIAMETER)

    focal_grid = make_focal_grid(
        q=4, num_airy=30,
        spatial_resolution=WAVELENGTH_WFS / TELESCOPE_DIAMETER)
    camera = NoiselessDetector(focal_grid)

    # Reference: flat wavefront — run through SH-WFS to get output grid metadata
    wf_ref = Wavefront(aperture, WAVELENGTH_WFS)
    wf_ref.total_power = flux_per_frame
    camera.integrate(shwfs(magnifier(wf_ref)), 1)
    image_ref_field = camera.read_out()               # HCIPy Field (keep for estimator)
    ref_image_np    = np.array(image_ref_field, dtype=np.float32)  # (N_det,) for saving

    # Grid metadata comes from the SH-WFS output field (NOT from focal_grid)
    # The camera accumulates the SHWFS output field as-is on its own 128×128 grid
    det_grid   = image_ref_field.grid
    N_det      = det_grid.size                                    # 16384 = 128×128
    det_shape  = [int(det_grid.dims[1]), int(det_grid.dims[0])]  # [H, W]
    coords_x_np = np.array(det_grid.x, dtype=np.float32)         # (N_det,) focal-plane x [m]
    coords_y_np = np.array(det_grid.y, dtype=np.float32)         # (N_det,) focal-plane y [m]
    print(f"Detector grid: {det_shape[0]}×{det_shape[1]} = {N_det} px")

    # WFS estimator (defines valid subapertures)
    shwfse = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid, shwfs.micro_lens_array.mla_index)

    fluxes     = ndimage.sum(image_ref_field, shwfse.mla_index,
                             shwfse.estimation_subapertures)
    flux_limit = fluxes.max() * 0.5
    valid_mask = shwfs.mla_grid.zeros(dtype='bool')
    valid_mask[shwfse.estimation_subapertures[fluxes > flux_limit]] = True
    shwfse = ShackHartmannWavefrontSensorEstimator(
        shwfs.mla_grid, shwfs.micro_lens_array.mla_index, valid_mask)

    # Validate slope ordering by extracting reference slopes (used for response matrix)
    slopes_ref_field = shwfse.estimate([image_ref_field])
    N_slopes = int(np.array(slopes_ref_field).ravel().size)
    N_subaps = N_slopes // 2
    print(f"Valid subapertures: {N_subaps},  slopes: {N_slopes}")

    # Build subaperture rank map: pixel → rank in estimation_subapertures (-1 = invalid)
    # shwfse.mla_index maps each detector pixel to its nearest lenslet index (size N_det)
    # We re-index so rank j corresponds to the j-th valid subaperture; ordering must
    # match HCIPy's estimate() output so slopes align with the response matrix.
    shwfse_mla_np  = np.array(shwfse.mla_index, dtype=np.int32)   # (N_det,)
    subap_rank_map = np.full(N_det, -1, dtype=np.int32)
    for rank, subap_idx in enumerate(shwfse.estimation_subapertures):
        subap_rank_map[shwfse_mla_np == int(subap_idx)] = rank
    print(f"Subaperture rank map: {(subap_rank_map >= 0).sum()} valid pixels")

    # ------------------------------------------------------------------
    # 3. Deformable mirror (disk-harmonic basis)
    # ------------------------------------------------------------------
    dm_modes_hcipy = make_disk_harmonic_basis(
        pupil_grid, NUM_MODES, TELESCOPE_DIAMETER, 'neumann')
    dm_modes_hcipy = ModeBasis(
        [mode / np.ptp(mode) for mode in dm_modes_hcipy], pupil_grid)
    dm = DeformableMirror(dm_modes_hcipy)

    dm_modes_np = np.array(
        [np.array(m, dtype=np.float32) for m in dm_modes_hcipy],
        dtype=np.float32)   # (N_modes, N_pupil_px)
    print(f"DM modes: {dm_modes_np.shape}")

    # ------------------------------------------------------------------
    # 4. Response matrix (push-pull calibration)
    # ------------------------------------------------------------------
    print("Building response matrix ...")
    probe_amp     = PROBE_AMP_WL_FRAC * WAVELENGTH_WFS
    response_rows = []
    wf_cal        = Wavefront(aperture, WAVELENGTH_WFS)
    wf_cal.total_power = flux_per_frame

    for i in range(NUM_MODES):
        slope_diff = 0
        for amp in [-probe_amp, probe_amp]:
            dm.flatten()
            dm.actuators[i] = amp
            cam_wf = shwfs(magnifier(dm(wf_cal)))
            camera.integrate(cam_wf, 1)
            img_field = camera.read_out()
            s         = shwfse.estimate([img_field])
            slope_diff += amp * np.array(s, dtype=np.float32)
        response_rows.append(
            (slope_diff / np.var([-probe_amp, probe_amp])).ravel())
        if (i + 1) % 50 == 0:
            print(f"  mode {i+1}/{NUM_MODES}")

    dm.flatten()
    response_matrix_np = np.array(response_rows, dtype=np.float32).T
    # (N_slopes, N_modes)
    print(f"Response matrix: {response_matrix_np.shape}")

    # ------------------------------------------------------------------
    # 5. Generate N_levels static wavefronts — save raw WFS images
    # ------------------------------------------------------------------
    print(f"\nGenerating wavefronts: WFE = {WFE_LEVELS_NM} nm ...")
    N_levels = len(WFE_LEVELS_NM)
    mask     = aperture_np > 0.5

    H_det, W_det   = det_shape
    wfs_images_all = np.zeros((N_levels, N_det),    dtype=np.float32)
    wf_phases_all  = np.zeros((N_levels, N_pupil_px), dtype=np.float32)

    wf_probe = Wavefront(aperture, WAVELENGTH_WFS)
    wf_probe.total_power = flux_per_frame

    for i, wfe_nm in enumerate(WFE_LEVELS_NM):
        # Independent random DM mode coefficients per level
        rng_i  = np.random.default_rng(100 + i)
        coeffs = rng_i.normal(0, 1, NUM_MODES).astype(np.float64)

        # DM phase [rad at lambda_wfs]: phi = 4pi * surface / lambda
        surface = dm_modes_np.T.astype(np.float64) @ coeffs   # (N_pupil_px,)
        phi_raw = 4.0 * np.pi * surface / WAVELENGTH_WFS       # [rad]

        # Scale to target WFE
        rms_raw        = float(np.std(phi_raw[mask]))
        target_rms_rad = 2.0 * np.pi * wfe_nm * 1e-9 / WAVELENGTH_WFS
        scale          = target_rms_rad / rms_raw
        coeffs_s       = (coeffs * scale).astype(np.float32)

        # Final phase map
        surface_s = dm_modes_np.T @ coeffs_s
        phi_s     = (4.0 * np.pi * surface_s / WAVELENGTH_WFS).astype(np.float32)
        actual_nm = float(np.std(phi_s[mask])) * WAVELENGTH_WFS / (2.0 * np.pi) * 1e9
        print(f"  Level {i+1}: target={wfe_nm} nm,  actual={actual_nm:.2f} nm")

        # Apply via DM, get raw WFS image with Poisson noise
        dm.flatten()
        dm.actuators = coeffs_s
        wf_dm     = dm(wf_probe)
        camera.integrate(shwfs(magnifier(wf_dm)), 1)
        img_field = camera.read_out()
        img_noisy = np.array(large_poisson(img_field), dtype=np.float32)

        wfs_images_all[i] = img_noisy
        wf_phases_all[i]  = phi_s

    dm.flatten()

    # ------------------------------------------------------------------
    # 6. Save data
    # ------------------------------------------------------------------
    np.savez(
        os.path.join(DATA_DIR, 'raw_data.npz'),
        response_matrix   = response_matrix_np[np.newaxis],                            # (1, N_slopes, N_modes)
        wfs_images        = wfs_images_all.reshape(N_levels, H_det, W_det)[np.newaxis], # (1, N_levels, H, W)
        ref_image         = ref_image_np.reshape(H_det, W_det)[np.newaxis],            # (1, H, W)
        detector_coords_x = coords_x_np.reshape(H_det, W_det)[np.newaxis],            # (1, H, W)
        detector_coords_y = coords_y_np.reshape(H_det, W_det)[np.newaxis],            # (1, H, W)
        subap_map         = subap_rank_map.reshape(H_det, W_det)[np.newaxis],          # (1, H, W)
        dm_modes          = dm_modes_np[np.newaxis],                                   # (1, N_modes, N_pupil_px)
        aperture          = aperture_np[np.newaxis],                                   # (1, N_pupil_px)
    )
    print("\nSaved raw_data.npz")

    np.savez(
        os.path.join(DATA_DIR, 'ground_truth.npz'),
        wavefront_phases = wf_phases_all[np.newaxis],             # (1, N_levels, N_pupil_px)
    )
    print("Saved ground_truth.npz")

    meta = {
        "telescope": {
            "diameter_m":        TELESCOPE_DIAMETER,
            "central_obs_m":     CENTRAL_OBS,
            "spider_width_m":    SPIDER_WIDTH,
            "n_spiders":         4
        },
        "wavefront_sensor": {
            "type":              "Shack-Hartmann",
            "n_lenslets":        NUM_LENSLETS,
            "f_number":          F_NUMBER,
            "sh_diameter_m":     SH_DIAMETER,
            "wavelength_wfs_m":  WAVELENGTH_WFS,
            "n_valid_subaps":    int(N_subaps),
            "n_slopes":          int(N_slopes),
            "n_det_pixels":      int(N_det),
            "det_image_shape":   det_shape
        },
        "deformable_mirror": {
            "n_modes":           NUM_MODES,
            "mode_basis":        "disk_harmonics_neumann"
        },
        "simulation": {
            "n_pupil_pixels":    NUM_PUPIL_PIXELS,
            "stellar_magnitude": STELLAR_MAGNITUDE,
            "photons_per_frame": int(flux_per_frame)
        },
        "wfe_levels_nm": WFE_LEVELS_NM
    }
    with open(os.path.join(DATA_DIR, 'meta_data.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved meta_data.json")

    print(f"\n=== Done ===  wfs_images ({N_levels},{H_det},{W_det}),  "
          f"wavefront_phases {wf_phases_all.shape}")


if __name__ == '__main__':
    main()
