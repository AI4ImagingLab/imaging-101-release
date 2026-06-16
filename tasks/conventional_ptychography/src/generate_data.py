"""
Generate synthetic conventional ptychography (CP) dataset.

Simulates a focused-probe x-ray/visible-light CP experiment:
  - Complex spiral object with phase and amplitude structure
  - Focused Gaussian-apodized probe (simulated through a lens)
  - Non-uniform Fermat scan grid (100 positions)
  - Far-field (Fraunhofer) diffraction data with Poisson noise

Output: data/simu.hdf5  (PtyLab-compatible HDF5 format)

Physical setup (matches PtyLab simulateData.py):
  - wavelength = 632.8 nm
  - detector: 128 × 128 pixels, pixel size 140 μm (effective at binning 4)
  - sample–detector distance zo = 50 mm
"""

import os
import numpy as np
import h5py
from scipy.signal import convolve2d
from .utils import circ, gaussian2D, cart2pol, fft2c, aspw, GenerateNonUniformFermat


def generate_probe(wavelength, zo, Nd, dxd, f=8e-3):
    """
    Generate a focused probe wavefield at the sample plane.

    The probe is formed by imaging an aperture through a thin lens:
    - A circular pinhole is propagated by angular spectrum (2f) to a lens
    - The lens applies a quadratic phase (focus) and a smooth aperture
    - The focused beam is propagated another 2f to the sample plane

    Parameters
    ----------
    wavelength : float
        Illumination wavelength [m].
    zo : float
        Sample-to-detector distance [m].
    Nd : int
        Number of detector pixels (square).
    dxd : float
        Detector pixel size [m].
    f : float
        Focal length of the focusing lens [m].

    Returns
    -------
    probe : ndarray, shape (Nd, Nd), complex128
        Complex probe wavefield in the sample plane.
    dxp : float
        Probe pixel size [m] (= wavelength * zo / (Nd * dxd)).
    """
    Ld = Nd * dxd
    dxp = wavelength * zo / Ld
    Np = Nd
    Lp = dxp * Np
    xp = np.arange(-Np // 2, Np // 2) * dxp
    Xp, Yp = np.meshgrid(xp, xp)

    # circular pinhole at aperture plane
    pinhole = circ(Xp, Yp, Lp / 2)
    pinhole = convolve2d(pinhole, gaussian2D(5, 1).astype(np.float32), mode="same")

    # propagate pinhole → lens plane (distance 2f)
    beam_at_lens = aspw(pinhole, 2 * f, wavelength, Lp)[0]

    # apply lens phase (quadratic) and soft aperture
    aperture = circ(Xp, Yp, 3 * Lp / 4)
    aperture = convolve2d(aperture, gaussian2D(5, 3).astype(np.float32), mode="same")
    beam_at_lens = (
        beam_at_lens
        * np.exp(-1j * 2 * np.pi / wavelength * (Xp ** 2 + Yp ** 2) / (2 * f))
        * aperture
    )

    # propagate lens → sample plane (distance 2f)
    probe = aspw(beam_at_lens, 2 * f, wavelength, Lp)[0]
    return probe, dxp


def generate_usaf_object(No: int, dxp: float, groups=(4, 5, 6, 7), phi_max: float = np.pi / 2) -> np.ndarray:
    """
    Generate a USAF 1951 phase test object.

    Amplitude = 1.0 everywhere; phase = phi_max where USAF bars are present,
    0 on background.  This is a pure-phase object (unit-amplitude transmission).

    Each USAF element (group G, element E) has:
      - 3 vertical bars on the left,  each bar_width × (5 × bar_width)
      - 3 horizontal bars on the right, each (5 × bar_width) × bar_width
    where bar_width = round(period / 2 / dxp) pixels
    and period = 1 / 2^(G + (E-1)/6)  mm.

    Parameters
    ----------
    No : int
        Object array size (square).
    dxp : float
        Object pixel size [m].
    groups : tuple of int
        USAF groups to render (higher group = finer features).
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
    group_widths, group_heights = [], []
    for G in groups:
        bw1 = period_to_bw(G, 1)
        g_w = 10 * bw1          # column width fixed to E1's scale
        g_h = 0
        for E in range(1, 7):
            bw = period_to_bw(G, E)
            g_h += 5 * bw
            if E < 6:
                g_h += max(1, bw // 2)   # inter-element gap
        group_widths.append(g_w)
        group_heights.append(g_h)

    inter_group_gap = [max(1, period_to_bw(G, 1) // 2) for G in groups]
    total_w = sum(group_widths) + sum(inter_group_gap[:-1])
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

        curr_c += group_widths[gi] + inter_group_gap[gi]

    obj = np.exp(1j * phi_max * mask)
    return obj.astype(np.complex64)



def generate_scan_grid(Np, dxp, num_points=100, radius=150):
    """
    Generate a non-uniform Fermat scan grid.

    The Fermat spiral provides quasi-random, aperiodic scan positions with
    high overlap uniformity, which is important for robust ptychographic
    convergence.

    The object size No and positions are computed to exactly match PtyLab's
    CPM No auto-detection logic (Reconstruction.py initializeSettings):
        No = max(Np, max(range_pixels_per_axis) + 2*Np)   (rounded to even)
        positions = round(encoder / dxp) + No // 2 - Np // 2

    Parameters
    ----------
    Np : int
        Probe array size (pixels) = Nd (detector size).
    dxp : float
        Pixel size [m].
    num_points : int
        Number of scan positions.
    radius : int
        Radius of scan grid in pixels.

    Returns
    -------
    positions : ndarray, shape (num_points, 2), int
        Scan positions in pixel coordinates (row, col) for upper-left corner
        of each probe window.
    encoder : ndarray, shape (num_points, 2), float
        Scan positions in physical coordinates [m] (row, col).
    No : int
        Object array size derived from actual encoder range (matches PtyLab).
    """
    R, C = GenerateNonUniformFermat(num_points, radius=radius, power=1)
    encoder = np.vstack((R * dxp, C * dxp)).T
    pix_raw = np.round(encoder / dxp).astype(int)

    # Mirror PtyLab's No auto-detection (Reconstruction.py initializeSettings):
    range_pix = np.max(pix_raw, axis=0) - np.min(pix_raw, axis=0)
    No = int(np.max([Np, np.max(range_pix) + 2 * Np]))
    if No % 2 == 1:
        No += 1

    positions = (pix_raw + No // 2 - Np // 2).astype(int)
    return positions, encoder, No


def generate_ptychogram(obj, probe, positions, Nd, bit_depth=14, seed=42):
    """
    Simulate the ptychographic diffraction dataset.

    For each scan position j:
      1. Extract object patch O_j of size (Np, Np)
      2. Compute exit wave: ψ_j = P * O_j
      3. Propagate to detector (Fraunhofer): Ψ_j = FT{ψ_j}
      4. Record intensity: I_j = |Ψ_j|²

    Poisson noise is added to model shot noise in photon-counting detectors.

    Parameters
    ----------
    obj : ndarray, shape (No, No), complex128
        Complex object.
    probe : ndarray, shape (Np, Np), complex128
        Complex probe.
    positions : ndarray, shape (J, 2), int
        Scan positions (row, col).
    Nd : int
        Detector size (= Np).
    bit_depth : int
        Detector dynamic range: max counts = 2^bit_depth.
    seed : int
        Random seed for Poisson noise.

    Returns
    -------
    ptychogram : ndarray, shape (J, Nd, Nd), float32
        Simulated diffraction intensities (with Poisson noise).
    """
    rng = np.random.default_rng(seed)
    Np = probe.shape[-1]
    num_frames = len(positions)
    ptychogram = np.zeros((num_frames, Nd, Nd), dtype=np.float32)

    max_counts = 2 ** bit_depth
    for j in range(num_frames):
        row, col = positions[j]
        sy = slice(row, row + Np)
        sx = slice(col, col + Np)
        obj_patch = obj[sy, sx].copy()
        esw = obj_patch * probe
        ESW = fft2c(esw)
        ptychogram[j] = np.abs(ESW) ** 2

    # normalize to detector dynamic range
    ptychogram = ptychogram / ptychogram.max() * max_counts
    # add Poisson noise
    ptychogram = ptychogram + rng.poisson(ptychogram).astype(np.float32)
    ptychogram = np.clip(ptychogram, 0, None)
    return ptychogram


def save_dataset(filepath, ptychogram, encoder, wavelength, zo, dxd, Nd,
                 No, entrancePupilDiameter):
    """
    Save the CP dataset as a PtyLab-compatible HDF5 file.

    The HDF5 schema matches PtyLab's ExperimentalData.loadData() requirements:
    - ptychogram : (J, Nd, Nd) float32
    - encoder    : (J, 2) float64  [m]
    - wavelength : scalar [m]
    - zo         : scalar [m]
    - dxd        : scalar [m]
    - Nd, No, binningFactor, entrancePupilDiameter, orientation

    Parameters
    ----------
    filepath : str or Path
        Output HDF5 file path.
    ptychogram : ndarray
        Diffraction data.
    encoder : ndarray
        Scan positions [m].
    wavelength, zo, dxd : float
        Physical parameters.
    Nd, No : int
        Detector and object sizes.
    entrancePupilDiameter : float
        Estimated probe diameter [m] (used for initial probe guess).
    """
    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")
        hf.create_dataset("encoder", data=encoder, dtype="f")
        hf.create_dataset("binningFactor", data=1, dtype="i")
        hf.create_dataset("dxd", data=(dxd,), dtype="f")
        hf.create_dataset("Nd", data=(Nd,), dtype="i")
        hf.create_dataset("No", data=(No,), dtype="i")
        hf.create_dataset("zo", data=(zo,), dtype="f")
        hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
        hf.create_dataset("entrancePupilDiameter", data=(entrancePupilDiameter,), dtype="f")
    print(f"Saved CP dataset to {filepath}")
    print(f"  ptychogram shape: {ptychogram.shape}, dtype: {ptychogram.dtype}")
    print(f"  num scan positions: {len(encoder)}")


def main(output_path=None):
    """Generate and save the CP simulation dataset."""
    import json
    from pathlib import Path

    task_dir = Path(__file__).parent.parent
    if output_path is None:
        output_path = task_dir / "data" / "raw_data.npz"

    # ---------- physical parameters ----------
    wavelength = 632.8e-9   # [m] He-Ne laser
    zo = 5e-2               # [m] sample–detector distance
    Nd = 128                # detector pixels (2^7)
    dxd = (2048 / Nd) * 4.5e-6   # effective pixel size [m] (binned from 4.5 μm sensor)

    print("Generating CP probe...")
    probe, dxp = generate_probe(wavelength, zo, Nd, dxd)

    # estimate probe (beam) diameter from second moment
    Lp = Nd * dxp
    xp = np.arange(-Nd // 2, Nd // 2) * dxp
    Xp, Yp = np.meshgrid(xp, xp)
    beam_size = (
        np.sqrt(np.sum((Xp**2 + Yp**2) * np.abs(probe)**2) / np.sum(np.abs(probe)**2))
        * 2.355  # FWHM conversion
    )
    print(f"  probe FWHM diameter: {beam_size*1e6:.1f} μm")

    print("Generating scan grid...")
    positions, encoder, No = generate_scan_grid(Nd, dxp, num_points=100, radius=150)
    print(f"  No={No}, scan range: rows [{positions[:,0].min()}, {positions[:,0].max()}]")

    print("Generating CP object (USAF 1951 phase chart)...")
    obj = generate_usaf_object(No, dxp)

    print("Simulating diffraction patterns...")
    ptychogram = generate_ptychogram(obj, probe, positions, Nd)
    print(f"  ptychogram: {ptychogram.shape}, max={ptychogram.max():.0f}")

    # save raw_data.npz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), ptychogram=ptychogram, encoder=encoder.astype(np.float32))
    print(f"Saved raw_data.npz to {output_path}")
    print(f"  ptychogram shape: {ptychogram.shape}, dtype: {ptychogram.dtype}")
    print(f"  num scan positions: {len(encoder)}")

    # save ground_truth.npz
    gt_path = output_path.parent / "ground_truth.npz"
    np.savez(str(gt_path), object=obj)
    print(f"Saved ground_truth.npz ({obj.shape})")

    # save meta_data.json
    meta = {
        "wavelength_m": wavelength,
        "zo_m": zo,
        "Nd": Nd,
        "dxd_m": float(dxd),
        "No": No,
        "dxp_m": float(dxp),
        "entrance_pupil_diameter_m": float(beam_size),
        "num_scan_positions": 100,
        "bit_depth": 14,
        "propagator": "Fraunhofer",
        "operation_mode": "CPM",
        "description": "Synthetic CP dataset: focused probe, USAF 1951 phase object (phi_max=pi/2), Fermat scan grid",
    }
    meta_path = task_dir / "data" / "meta_data.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta_data.json to {meta_path}")
    return output_path


if __name__ == "__main__":
    main()
