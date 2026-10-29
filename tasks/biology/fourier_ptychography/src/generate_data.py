"""
Generate synthetic Fourier ptychography (FP) dataset.

Simulates a visible-light FPM experiment:
  - High-resolution USAF 1951 phase object (amplitude=1, binary phase)
  - LED array (11×11 = 121 LEDs) at known angles
  - Microscope with NA=0.1, magnification=4, λ=625nm
  - Low-resolution bright- and dark-field images with Poisson noise

FPM forward model (matches PtyLab's mqNewton engine):
  For LED at position (ux, uy) at distance z_led:
  - k-space shift in pixels: Δk = -(Nd * dxp / λ) * (ux, uy) / dist
    where dist = sqrt(ux²+uy²+z_led²)  [same formula as PtyLab Reconstruction.positions]
  - Window start in No×No spectrum: r0 = No//2 - Nd//2 + Δk_row
  - Low-res image: I_j = |FFT{ O_spectrum[r0:r0+Nd, c0:c0+Nd] · P̃ }|²
    (FFT not IFFT, matching PtyLab's Fraunhofer propagator for FPM)

Output: data/raw_data.npz, data/ground_truth.npz, data/meta_data.json
"""

import os
import numpy as np
import h5py
from .utils import fft2c


def generate_usaf_object(No: int, dxp: float, groups=(5, 6, 7, 8), phi_max: float = np.pi / 2) -> np.ndarray:
    """
    Generate a USAF 1951 phase test object for Fourier ptychography.

    Amplitude = 1.0 everywhere; phase = phi_max where USAF bars are present,
    0 on background.  This is a pure-phase object suitable for FPM benchmarking.

    Each USAF element (group G, element E) consists of:
      - 3 vertical bars on the left:  each bar_width × (5 × bar_width)
      - 3 horizontal bars on the right: each (5 × bar_width) × bar_width
    where bar_width = round(period / 2 / dxp) pixels
    and period = 1 / 2^(G + (E-1)/6)  mm.

    For FPM with dxp=1.625 μm, Groups 5–8 span periods from 31 μm (easily
    resolved) down to 3.9 μm (near the diffraction limit at NA_synthetic≈0.27).

    Parameters
    ----------
    No : int
        Object array size (square).
    dxp : float
        Object pixel size [m].
    groups : tuple of int
        USAF groups to render (higher = finer features).
    phi_max : float
        Phase value [rad] for bar regions (background = 0).

    Returns
    -------
    obj : ndarray, shape (No, No), complex64
        Complex phase object: exp(i * phi_max * usaf_mask).
    """
    dxp_mm = dxp * 1e3  # m → mm

    def period_to_bw(G, E):
        """Bar width in pixels for group G, element E."""
        T_mm = 1.0 / (2 ** (G + (E - 1) / 6))
        return max(1, round(T_mm / 2 / dxp_mm))

    # ── Compute total chart dimensions ────────────────────────────────────
    group_widths, group_heights, inter_group_gaps = [], [], []
    for G in groups:
        bw1 = period_to_bw(G, 1)
        g_w = 10 * bw1
        g_h = 0
        for E in range(1, 7):
            bw = period_to_bw(G, E)
            g_h += 5 * bw
            if E < 6:
                g_h += max(1, bw // 2)
        group_widths.append(g_w)
        group_heights.append(g_h)
        inter_group_gaps.append(max(1, bw1 // 2))

    total_w = sum(group_widths) + sum(inter_group_gaps[:-1])
    total_h = max(group_heights)

    # ── Center chart in the object array ──────────────────────────────────
    r0 = max(0, No // 2 - total_h // 2)
    c0 = max(0, No // 2 - total_w // 2)

    mask = np.zeros((No, No), dtype=np.float32)

    curr_c = c0
    for gi, G in enumerate(groups):
        bw1 = period_to_bw(G, 1)
        g_h = group_heights[gi]

        curr_r = r0 + (total_h - g_h) // 2   # vertically center each group

        for E in range(1, 7):
            bw = period_to_bw(G, E)
            elem_h = 5 * bw

            r, c = curr_r, curr_c

            # 3 vertical bars
            for i in range(3):
                cs, ce = c + i * 2 * bw, c + i * 2 * bw + bw
                re = r + elem_h
                if re <= No and ce <= No:
                    mask[r:re, cs:ce] = 1.0

            # 3 horizontal bars (right of vertical section)
            h_c0 = c + 5 * bw
            for i in range(3):
                rs, re = r + i * 2 * bw, r + i * 2 * bw + bw
                cs, ce = h_c0, h_c0 + 5 * bw
                if re <= No and ce <= No:
                    mask[rs:re, cs:ce] = 1.0

            curr_r += elem_h + max(1, bw // 2)

        curr_c += group_widths[gi] + inter_group_gaps[gi]

    obj = np.exp(1j * phi_max * mask)
    return obj.astype(np.complex64)



def generate_led_array(
    n_leds_side: int = 11,
    led_pitch: float = 4e-3,
    z_led: float = 60e-3,
) -> tuple:
    """
    Generate a square LED array centered on the optical axis.

    Parameters
    ----------
    n_leds_side : int
        Number of LEDs per side (total = n_leds_side²).
    led_pitch : float
        LED spacing [m].
    z_led : float
        LED-to-sample distance [m].

    Returns
    -------
    encoder : ndarray, shape (J, 2), float
        LED x/y positions [m] from optical axis (row, col convention).
    led_angles : ndarray, shape (J, 2), float
        Illumination angles [rad] for each LED.
    """
    idx = np.arange(n_leds_side) - n_leds_side // 2
    ux, uy = np.meshgrid(idx * led_pitch, idx * led_pitch)
    # encoder: (row_position, col_position) = (y, x) in meters
    encoder = np.column_stack([uy.ravel(), ux.ravel()])
    led_angles = encoder / z_led
    return encoder, led_angles


def compute_pupil(Nd: int, dxp: float, wavelength: float, NA: float) -> np.ndarray:
    """
    Compute the binary pupil function (bandpass filter) in k-space.

    The pupil is a circle in the Fourier plane of the object, with radius:
        r_pupil = NA * Nd * dxp / wavelength   [pixels]

    Parameters
    ----------
    Nd : int
        Detector / pupil size in pixels.
    dxp : float
        Object pixel size [m] (= dxd / magnification).
    wavelength : float
        Illumination wavelength [m].
    NA : float
        Numerical aperture of the imaging lens.

    Returns
    -------
    pupil : ndarray, shape (Nd, Nd), float
        Binary pupil mask (1 inside NA circle, 0 outside).
    """
    # Spatial frequency coordinates (cycles/pixel)
    fx = np.fft.fftshift(np.fft.fftfreq(Nd))
    FX, FY = np.meshgrid(fx, fx)
    # Convert radius: f_cutoff = NA / (wavelength / dxp) = NA * dxp / wavelength
    f_cutoff = NA * dxp / wavelength
    pupil = ((FX**2 + FY**2) <= f_cutoff**2).astype(np.float32)
    return pupil


def simulate_fpm_images(
    obj: np.ndarray,
    pupil: np.ndarray,
    encoder: np.ndarray,
    z_led: float,
    wavelength: float,
    dxp: float,
    Nd: int,
    bit_depth: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate the FPM low-resolution image stack.

    For each LED at (ux, uy), the k-shift in pixel units is:
        (Δkx, Δky) = (ux * Nd * dxp) / (wavelength * z_led)

    The low-resolution image is computed as:
        I_j(r) = |IFFT{P̃(q) · roll(Õ, shift)}|²

    where Õ = FT{O} is the object spectrum and rolling implements the shift.

    Parameters
    ----------
    obj : ndarray, shape (No, No), complex
        High-resolution complex object.
    pupil : ndarray, shape (Nd, Nd), float
        Pupil function in k-space (fftshift convention).
    encoder : ndarray, shape (J, 2)
        LED positions [m].
    z_led : float
        LED-to-sample distance [m].
    wavelength : float
        Illumination wavelength [m].
    dxp : float
        Object pixel size [m].
    Nd : int
        Low-resolution image size (= detector pixels).
    bit_depth : int
        Detector dynamic range.
    seed : int
        RNG seed for Poisson noise.

    Returns
    -------
    ptychogram : ndarray, shape (J, Nd, Nd), float32
        Low-resolution FPM images with noise.
    """
    rng = np.random.default_rng(seed)
    No = obj.shape[0]
    num_leds = len(encoder)

    # Compute high-resolution object spectrum (DC at center via fftshift)
    Obj_spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))

    # The high-res object spectrum is No × No, pupil is Nd × Nd
    # We extract the Nd × Nd sub-region centered at the shifted position
    center = No // 2

    ptychogram = np.zeros((num_leds, Nd, Nd), dtype=np.float32)
    max_intensity = 0.0

    # PtyLab positions formula: conv = -(1/λ) * dxp * Nd
    # k-shift = conv * (uy, ux) / sqrt(uy²+ux²+z_led²)
    # window start: r0 = No//2 - Nd//2 + dk_row  (matches Reconstruction.positions)
    conv = -(1.0 / wavelength) * dxp * Nd

    for j, (uy, ux) in enumerate(encoder):
        dist = np.sqrt(uy**2 + ux**2 + z_led**2)
        dk_row = int(round(conv * uy / dist))
        dk_col = int(round(conv * ux / dist))

        # Extract Nd × Nd subregion of spectrum centered at (center + dk)
        r0 = center - Nd // 2 + dk_row
        c0 = center - Nd // 2 + dk_col
        r1, c1 = r0 + Nd, c0 + Nd

        # boundary check: skip if out of range
        if r0 < 0 or r1 > No or c0 < 0 or c1 > No:
            continue

        sub_spectrum = Obj_spectrum[r0:r1, c0:c1]

        # apply pupil (already in fftshift convention)
        filtered = sub_spectrum * pupil

        # FFT → low-res image (matches PtyLab's Fraunhofer propagator: ESW = fft2c(esw))
        lr_field = fft2c(filtered)
        lr_image = np.abs(lr_field) ** 2
        ptychogram[j] = lr_image.astype(np.float32)
        if lr_image.max() > max_intensity:
            max_intensity = lr_image.max()

    # normalize to bit depth
    max_counts = 2 ** bit_depth
    ptychogram = ptychogram / (max_intensity + 1e-12) * max_counts

    # Poisson noise (shot noise: Y ~ Poisson(lambda), E[Y]=lambda, Var[Y]=lambda)
    ptychogram_noisy = rng.poisson(ptychogram).astype(np.float32)
    ptychogram_noisy = np.clip(ptychogram_noisy, 0, None)
    return ptychogram_noisy


def save_fpm_dataset(
    filepath: str,
    ptychogram: np.ndarray,
    encoder: np.ndarray,
    wavelength: float,
    z_led: float,
    magnification: float,
    dxd: float,
    NA: float,
    No: int = None,
):
    """
    Save the FPM dataset as a PtyLab-compatible HDF5 file.

    FPM HDF5 schema (ExperimentalData FPM mode):
    - ptychogram   : (J, Nd, Nd) float32   – low-resolution images
    - encoder      : (J, 2) float64        – LED positions [m]
    - wavelength   : scalar [m]
    - zled         : scalar [m]
    - magnification: scalar
    - dxd          : scalar [m]
    - NA           : scalar (optional, helps pupil initialization)
    - entrancePupilDiameter: scalar [m]

    Parameters
    ----------
    filepath : str
    ptychogram, encoder : ndarray
    wavelength, z_led, magnification, dxd, NA : float
    No : int, optional
        High-resolution object size (if provided, stored in HDF5 for reference).
    """
    dxp = dxd / magnification
    Nd = ptychogram.shape[-1]
    # entrance pupil diameter = 2 * NA * f, but for FPM use dxp * Nd * NA / wavelength
    # PtyLab uses entrancePupilDiameter as initial pupil radius in pixels × dxp
    epd = 2 * NA * dxp * Nd / wavelength * wavelength  # = 2 * NA * Nd * dxp

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
        hf.create_dataset("zled", data=(z_led,), dtype="f")
        hf.create_dataset("magnification", data=(magnification,), dtype="f")
        hf.create_dataset("dxd", data=(dxd,), dtype="f")
        hf.create_dataset("NA", data=(NA,), dtype="f")
        hf.create_dataset("entrancePupilDiameter", data=(2 * NA * Nd * dxp,), dtype="f")
        hf.create_dataset("binningFactor", data=1, dtype="i")
        if No is not None:
            hf.create_dataset("No", data=(No,), dtype="i")

    print(f"Saved FPM dataset to {filepath}")
    print(f"  ptychogram shape: {ptychogram.shape}, dtype: {ptychogram.dtype}")
    print(f"  num LEDs: {len(encoder)}")


def main(output_path=None):
    """Generate and save the FPM simulation dataset."""
    import json
    from pathlib import Path

    task_dir = Path(__file__).parent.parent
    if output_path is None:
        output_path = task_dir / "data" / "raw_data.npz"

    # ---------- physical parameters ----------
    wavelength = 625e-9       # [m] red LED
    magnification = 4.0       # 4× microscope objective
    NA = 0.1                  # numerical aperture
    dxd = 6.5e-6              # [m] camera pixel size
    dxp = dxd / magnification  # [m] object pixel size = 1.625 μm
    Nd = 256                  # low-res image size (pixels)
    z_led = 60e-3             # [m] LED-to-sample distance
    n_leds_side = 11          # 11×11 LED array (most will fit within k-space bounds)
    led_pitch = 2e-3          # [m] LED spacing (2mm pitch ≈ 3.3 mrad per step)

    print(f"FPM parameters:")
    print(f"  λ={wavelength*1e9:.0f} nm, NA={NA}, M={magnification}")
    print(f"  Nd={Nd}, dxp={dxp*1e6:.2f} μm")
    print(f"  LED array: {n_leds_side}×{n_leds_side}, pitch={led_pitch*1e3:.0f} mm, z_led={z_led*1e3:.0f} mm")

    print("Generating LED array...")
    encoder, led_angles = generate_led_array(n_leds_side, led_pitch, z_led)

    # Compute k-shifts using the same formula as PtyLab (Reconstruction.positions for FPM):
    #   conv = -(1/λ) * dxp * Nd
    #   dk = round(conv * encoder / dist)
    dists = np.sqrt(encoder[:, 0]**2 + encoder[:, 1]**2 + z_led**2)
    conv = -(1.0 / wavelength) * dxp * Nd
    shifts_row = np.round(conv * encoder[:, 0] / dists).astype(int)
    shifts_col = np.round(conv * encoder[:, 1] / dists).astype(int)

    # Compute No using the EXACT same formula as PtyLab (Reconstruction.__init__):
    #   range_pixels = max(range_row, range_col) + 2*Np  (Np=Nd for FPM)
    #   No = max(Nd, range_pixels)  [rounded up to even]
    range_pixels = max(shifts_row.max() - shifts_row.min(),
                       shifts_col.max() - shifts_col.min()) + Nd * 2
    if range_pixels % 2 == 1:
        range_pixels += 1
    No = max(Nd, range_pixels)

    # Filter LEDs whose k-shift window fits within the No×No object spectrum
    max_shift_pix = (No - Nd) // 2 - 5
    valid = (np.abs(shifts_row) <= max_shift_pix) & (np.abs(shifts_col) <= max_shift_pix)
    encoder = encoder[valid]
    print(f"  {valid.sum()}/{len(valid)} LEDs with k-shift within object spectrum")
    print(f"  No={No} (computed from actual LED range, matching PtyLab formula)")

    print("Generating FPM object (USAF 1951 phase chart)...")
    obj = generate_usaf_object(No, dxp)

    print("Computing pupil...")
    pupil = compute_pupil(Nd, dxp, wavelength, NA)
    print(f"  pupil sum: {int(pupil.sum())} pixels (of {Nd*Nd})")

    print("Simulating FPM images...")
    ptychogram = simulate_fpm_images(
        obj, pupil, encoder, z_led, wavelength, dxp, Nd,
    )
    print(f"  ptychogram: {ptychogram.shape}, max={ptychogram.max():.0f}")

    # save raw_data.npz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), ptychogram=ptychogram, encoder=encoder.astype(np.float32))
    print(f"Saved raw_data.npz to {output_path}")
    print(f"  ptychogram shape: {ptychogram.shape}, dtype: {ptychogram.dtype}")
    print(f"  num LEDs: {len(encoder)}")

    # save ground_truth.npz
    gt_path = output_path.parent / "ground_truth.npz"
    np.savez(str(gt_path), object=obj)
    print(f"Saved ground_truth.npz ({obj.shape}) to {gt_path}")

    # save meta_data.json
    meta = {
        "wavelength_m": wavelength,
        "NA": NA,
        "magnification": magnification,
        "dxd_m": dxd,
        "dxp_m": dxp,
        "z_led_m": z_led,
        "led_pitch_m": led_pitch,
        "n_leds_side": n_leds_side,
        "num_leds": int(valid.sum()),
        "Nd": Nd,
        "No": int(No),
        "bit_depth": 10,
        "operation_mode": "FPM",
        "description": "Synthetic FPM dataset: USAF 1951 phase object (phi_max=pi/2), 11x11 LED array",
    }
    meta_path = task_dir / "data" / "meta_data.json"
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"Saved meta_data.json to {meta_path}")
    return output_path


if __name__ == "__main__":
    main()
