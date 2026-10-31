"""Forward model for Fourier Light Field Microscopy (FLFM).

Implements the optical geometry, micro-lens array lattice generation,
lens assignment by nearest-neighbour search, and the angular-sensitivity
(alpha) model needed for parallax-based depth estimation.

Extracted from: hexSMLFM / PySMLFM (TheLeeLab / Photometrics, Cambridge).
Reference: R. R. Sims et al., Optica 7, 1065 (2020).

Physics summary
---------------
The FLFM's MLA sits at the conjugate back-focal plane (BFP).  Each
micro-lens samples a different view angle (u, v) of the emission cone.
A molecule at 3D position (x0, y0, z) appears at image position
(x_k, y_k) through micro-lens k:

    x_k = x0 + u_k / rho  +  z * alpha_u(u_k, v_k)
    y_k = y0 + v_k / rho  +  z * alpha_v(u_k, v_k)

where rho = magnification / BFP_radius converts image-plane microns to
normalised pupil coordinates (u, v).

Alpha models
------------
LINEAR:           alpha = (u, v)
SPHERE:           alpha = -(NA/n)/sqrt(1 - rho^2*(NA/n)^2) * (u, v)
INTEGRATE_SPHERE: phase-averaged spherical model over each microlens aperture
"""

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Optical system
# ---------------------------------------------------------------------------

@dataclass
class FourierMicroscope:
    """Derived optical quantities for a Fourier Light Field Microscope.

    All lengths in microns unless noted.  Instantiate via build_microscope().

    Attributes:
        num_aperture:          Objective numerical aperture.
        mla_lens_pitch:        MLA pitch along X (microns).
        focal_length_mla:      MLA focal length (mm).
        focal_length_obj_lens: Objective focal length (mm).
        focal_length_tube_lens: Tube lens focal length (mm).
        focal_length_fourier_lens: Fourier relay focal length (mm).
        pixel_size_camera:     Camera pixel size (microns).
        ref_idx_immersion:     Immersion refractive index.
        ref_idx_medium:        Specimen/medium refractive index.

        bfp_radius:      Radius of conjugate BFP at MLA plane (microns).
        bfp_lens_count:  Number of micro-lenses across the BFP diameter.
        pixels_per_lens: Camera pixels per micro-lens.
        magnification:   Total magnification (sample → camera).
        pixel_size_sample: Pixel size in sample space (microns).
        rho_scaling:     Scale factor: image-plane microns → normalised pupil.
        xy_to_uv_scale:  = rho_scaling.
        mla_to_uv_scale: Converts MLA lattice spacing → pupil coords.
        mla_to_xy_scale: Converts MLA lattice spacing → image microns.
    """
    num_aperture: float
    mla_lens_pitch: float
    focal_length_mla: float
    focal_length_obj_lens: float
    focal_length_tube_lens: float
    focal_length_fourier_lens: float
    pixel_size_camera: float
    ref_idx_immersion: float
    ref_idx_medium: float

    # Derived (set by build_microscope)
    bfp_radius: float = 0.0
    bfp_lens_count: float = 0.0
    pixels_per_lens: float = 0.0
    magnification: float = 0.0
    pixel_size_sample: float = 0.0
    rho_scaling: float = 0.0
    xy_to_uv_scale: float = 0.0
    mla_to_uv_scale: float = 0.0
    mla_to_xy_scale: float = 0.0


def build_microscope(meta: dict) -> FourierMicroscope:
    """Construct a FourierMicroscope and compute all derived quantities.

    Derivations:
        bfp_radius      = 1e3 * NA * f_obj * (f_fourier / f_tube)  [µm]
        magnification   = (f_tube / f_obj) * (f_mla / f_fourier)
        pixel_size_sample = pixel_size_camera / magnification       [µm]
        rho_scaling     = magnification / bfp_radius
        bfp_lens_count  = 2 * bfp_radius / mla_lens_pitch
        mla_to_uv_scale = 2 / bfp_lens_count
        mla_to_xy_scale = mla_to_uv_scale / rho_scaling

    Args:
        meta: Dictionary from meta_data.json.

    Returns:
        FourierMicroscope with all fields populated.
    """
    NA   = meta["num_aperture"]
    p    = meta["mla_lens_pitch"]
    fmla = meta["focal_length_mla"]
    fobj = meta["focal_length_obj_lens"]
    ftub = meta["focal_length_tube_lens"]
    ffou = meta["focal_length_fourier_lens"]
    pix  = meta["pixel_size_camera"]
    ni   = meta["ref_idx_immersion"]
    nm   = meta["ref_idx_medium"]

    bfp_radius     = 1000.0 * NA * fobj * (ffou / ftub)
    magnification  = (ftub / fobj) * (fmla / ffou)
    bfp_lens_count = 2.0 * bfp_radius / p
    pixels_per_lens = p / pix
    pixel_size_sample = pix / magnification
    rho_scaling    = magnification / bfp_radius
    mla_to_uv_scale = 2.0 / bfp_lens_count
    mla_to_xy_scale = mla_to_uv_scale / rho_scaling

    return FourierMicroscope(
        num_aperture=NA, mla_lens_pitch=p, focal_length_mla=fmla,
        focal_length_obj_lens=fobj, focal_length_tube_lens=ftub,
        focal_length_fourier_lens=ffou, pixel_size_camera=pix,
        ref_idx_immersion=ni, ref_idx_medium=nm,
        bfp_radius=bfp_radius, bfp_lens_count=bfp_lens_count,
        pixels_per_lens=pixels_per_lens, magnification=magnification,
        pixel_size_sample=pixel_size_sample, rho_scaling=rho_scaling,
        xy_to_uv_scale=rho_scaling,
        mla_to_uv_scale=mla_to_uv_scale, mla_to_xy_scale=mla_to_xy_scale,
    )


# ---------------------------------------------------------------------------
# Micro-lens array
# ---------------------------------------------------------------------------

@dataclass
class MicroLensArray:
    """Hexagonal or square micro-lens array.

    Attributes:
        lattice_type: "HEXAGONAL" or "SQUARE".
        focal_length: MLA focal length (mm).
        lens_pitch:   MLA pitch along X (microns).
        optic_size:   Physical side length of the MLA optic (microns).
        centre:       Array centre in lattice-spacing units, shape (2,).
        lens_centres: All lens X,Y positions in lattice-spacing units, (K, 2).
    """
    lattice_type: str
    focal_length: float
    lens_pitch: float
    optic_size: float
    centre: npt.NDArray[float]
    lens_centres: npt.NDArray[float]


def build_mla(meta: dict) -> MicroLensArray:
    """Generate a MicroLensArray lattice and apply rotation from meta_data.

    The lattice is generated in the canonical orientation (0°), then rotated
    by meta["mla_rotation"] degrees so lens centres align with the camera data.

    Args:
        meta: Dictionary from meta_data.json.

    Returns:
        MicroLensArray with lens_centres rotated (and optionally offset).
    """
    lattice_type = meta["mla_type"].upper()
    focal_length = meta["focal_length_mla"]
    lens_pitch   = meta["mla_lens_pitch"]
    optic_size   = meta["mla_optic_size"]
    centre       = np.asarray(meta["mla_centre"], dtype=float)

    lens_centres = _generate_lattice(lattice_type, optic_size, lens_pitch)
    mla = MicroLensArray(
        lattice_type=lattice_type, focal_length=focal_length,
        lens_pitch=lens_pitch, optic_size=optic_size,
        centre=centre, lens_centres=lens_centres,
    )

    _rotate_lattice(mla, np.deg2rad(meta["mla_rotation"]))

    offset = np.asarray(meta.get("mla_offset", [0.0, 0.0]), dtype=float)
    if np.any(offset != 0.0):
        lfm = build_microscope(meta)
        _offset_lattice(mla, offset / lfm.mla_to_xy_scale)

    return mla


def _generate_lattice(lattice_type: str,
                      optic_size: float,
                      lens_pitch: float) -> npt.NDArray[float]:
    """Generate raw X,Y lens centre coordinates (in lattice-spacing units).

    For both lattice types the optic is assumed square; more centres than
    strictly needed are generated so that cropping after rotation is implicit
    (unused lenses are filtered later by BFP radius).
    """
    width = optic_size / lens_pitch
    marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)

    if lattice_type == "SQUARE":
        x, y = np.meshgrid(marks, marks)

    elif lattice_type == "HEXAGONAL":
        xx, yy = np.meshgrid(marks, marks)
        length = max(xx.shape)
        dx = np.tile([0.5, 0], [length, int(np.ceil(length / 2))])
        dx = dx[:length, :length]
        x = yy * np.sqrt(3) / 2
        y = xx + dx.T + 0.5

    else:
        raise ValueError(f"Unsupported lattice_type {lattice_type!r}. "
                         "Choose HEXAGONAL or SQUARE.")

    return np.column_stack((x.flatten("F"), y.flatten("F")))


def _rotate_lattice(mla: MicroLensArray, theta: float) -> None:
    """Rotate lens_centres around mla.centre by theta radians (in-place)."""
    x = mla.lens_centres[:, 0] - mla.centre[0]
    y = mla.lens_centres[:, 1] - mla.centre[1]
    mla.lens_centres[:, 0] = x * np.cos(theta) - y * np.sin(theta) + mla.centre[0]
    mla.lens_centres[:, 1] = x * np.sin(theta) + y * np.cos(theta) + mla.centre[1]


def _offset_lattice(mla: MicroLensArray, dxy: npt.NDArray[float]) -> None:
    """Shift lens_centres by dxy (in lattice-spacing units, in-place)."""
    mla.lens_centres += dxy


# ---------------------------------------------------------------------------
# Localisation data container and operations
# ---------------------------------------------------------------------------

class Localisations:
    """Container for 2D localisation data with light-field metadata.

    Attributes:
        locs_2d (N, 13): Full localisation array.
            [0]  frame
            [1]  U (normalised pupil coord of assigned lens)
            [2]  V
            [3]  X (microns)
            [4]  Y (microns)
            [5]  sigma_X (microns)
            [6]  sigma_Y (microns)
            [7]  intensity (photons)
            [8]  background (photons)
            [9]  precision (microns)
            [10] alpha_U
            [11] alpha_V
            [12] lens index (int)
        min_frame, max_frame: Frame range in the data.
        filtered_locs_2d: Subset of locs_2d after filtering (or None).
        corrected_locs_2d: XY-corrected copy of filtered_locs_2d (or None).
    """

    def __init__(self, locs_2d_csv: npt.NDArray[float]):
        self.min_frame = int(np.min(locs_2d_csv[:, 0]))
        self.max_frame = int(np.max(locs_2d_csv[:, 0]))

        self.locs_2d = np.zeros((locs_2d_csv.shape[0], 13))
        self.locs_2d[:, 0]    = locs_2d_csv[:, 0]   # frame
        self.locs_2d[:, 3:10] = locs_2d_csv[:, 1:8] # X,Y,sX,sY,I,bg,prec

        self.filtered_locs_2d = self.locs_2d.copy()
        self.corrected_locs_2d = None
        self.correction = None

    # ---- Lens assignment ------------------------------------------------

    def assign_to_lenses(self,
                         mla: MicroLensArray,
                         lfm: FourierMicroscope) -> None:
        """Map each localisation to its nearest micro-lens by kNN in (u,v).

        Converts (x, y) → normalised pupil coords (u, v) = (x, y) * rho_scaling,
        then finds the nearest lens centre.  Populates columns 1, 2 (U, V) and
        12 (lens index) in locs_2d.
        """
        xy = self.locs_2d[:, 3:5].copy() * lfm.xy_to_uv_scale

        lens_centres_uv = (mla.lens_centres - mla.centre) * lfm.mla_to_uv_scale

        knn = NearestNeighbors(n_neighbors=1).fit(lens_centres_uv)
        idx = knn.kneighbors(xy, return_distance=False)

        self.locs_2d[:, 1:3] = lens_centres_uv[idx, :][:, 0, :]
        self.locs_2d[:, 12]  = idx[:, 0]

        self.filtered_locs_2d = self.locs_2d.copy()
        self.corrected_locs_2d = None

    # ---- Filtering -------------------------------------------------------

    def filter_lenses(self,
                      mla: MicroLensArray,
                      lfm: FourierMicroscope) -> None:
        """Keep only localisations assigned to lenses inside the BFP circle."""
        radius_sq = (lfm.bfp_radius / mla.lens_pitch) ** 2
        centres   = mla.lens_centres - mla.centre
        dist_sq   = np.sum(centres ** 2, axis=1)
        valid_idx = np.nonzero(dist_sq < radius_sq)[0]

        keep = np.isin(self.filtered_locs_2d[:, 12], valid_idx)
        self.filtered_locs_2d = self.filtered_locs_2d[keep]

    def filter_rhos(self, rho_range: Tuple[float, float]) -> None:
        """Keep only localisations with pupil radius in [rho_min, rho_max]."""
        uv   = self.filtered_locs_2d[:, 1:3]
        rhos = np.sqrt(np.sum(uv ** 2, axis=1))
        keep = (rhos >= rho_range[0]) & (rhos <= rho_range[1])
        self.filtered_locs_2d = self.filtered_locs_2d[keep]

    def filter_spot_sizes(self, size_range: Tuple[float, float]) -> None:
        """Keep only localisations with sigma_X in [min, max] microns."""
        s    = self.filtered_locs_2d[:, 5]
        keep = (s >= size_range[0]) & (s <= size_range[1])
        self.filtered_locs_2d = self.filtered_locs_2d[keep]

    def filter_photons(self, ph_range: Tuple[float, float]) -> None:
        """Keep only localisations with photon count in [min, max]."""
        ph   = self.filtered_locs_2d[:, 7]
        keep = (ph >= ph_range[0]) & (ph <= ph_range[1])
        self.filtered_locs_2d = self.filtered_locs_2d[keep]

    # ---- Alpha model -------------------------------------------------------

    def compute_alpha_model(self,
                            lfm: FourierMicroscope,
                            model: str = "INTEGRATE_SPHERE",
                            worker_count: int = 1) -> None:
        """Compute angular sensitivity alpha(u,v) for each localisation.

        Alpha encodes how a unit depth change shifts the apparent lateral
        position of a molecule in a given view.

        Models:
            LINEAR:           alpha = (u, v)
            SPHERE:           alpha = -(NA/n) / sqrt(1 - rho^2*(NA/n)^2) * (u,v)
            INTEGRATE_SPHERE: phase-averaged sphere over each microlens aperture

        Results are stored in filtered_locs_2d[:, 10:12] (alpha_U, alpha_V).

        Args:
            lfm:          FourierMicroscope instance.
            model:        One of "LINEAR", "SPHERE", "INTEGRATE_SPHERE".
            worker_count: (unused; kept for API compatibility)
        """
        uv       = self.filtered_locs_2d[:, 1:3]
        alpha_uv = self.filtered_locs_2d[:, 10:12]
        na = lfm.num_aperture
        n  = lfm.ref_idx_medium

        if model == "LINEAR":
            alpha_uv[:] = uv

        elif model == "SPHERE":
            rho    = np.sqrt(np.sum(uv ** 2, axis=1))
            dr_sq  = 1.0 - rho * (na / n) ** 2
            dr_sq[dr_sq < 0.0] = np.nan
            phi    = -(na / n) / np.sqrt(dr_sq)
            alpha_uv[:] = uv * phi[:, np.newaxis]

        elif model == "INTEGRATE_SPHERE":
            uv_scaling = lfm.mla_to_uv_scale
            _phase_average_sphere(uv, uv_scaling, na, n, m=10, out=alpha_uv)

        else:
            raise ValueError(f"Unknown alpha model {model!r}. "
                             "Choose LINEAR, SPHERE, or INTEGRATE_SPHERE.")

    # ---- Aberration correction -------------------------------------------

    def correct_xy(self, correction: npt.NDArray[float]) -> None:
        """Apply per-view XY corrections to filtered_locs_2d.

        Subtracts the per-view mean fit error (dx, dy) from the X, Y
        coordinates of each localisation.  Result is stored in
        corrected_locs_2d.

        Args:
            correction: (V, 5) array — columns [0] U, [1] V, [2] dx, [3] dy,
                        [4] n_points (as returned by solvers.calculate_view_error).
        """
        self.corrected_locs_2d = self.filtered_locs_2d.copy()
        self.correction = correction.copy()

        u = self.corrected_locs_2d[:, 1]
        v = self.corrected_locs_2d[:, 2]
        x = self.corrected_locs_2d[:, 3]
        y = self.corrected_locs_2d[:, 4]

        for row in correction:
            idx = (u == row[0]) & (v == row[1])
            x[idx] -= row[2]
            y[idx] -= row[3]


# ---------------------------------------------------------------------------
# Phase-averaged sphere alpha model
# ---------------------------------------------------------------------------

def _phase_average_sphere(uv: npt.NDArray[float],
                           uv_scaling: float,
                           na: float,
                           n: float,
                           m: int,
                           out: npt.NDArray[float]) -> None:
    """Compute phase-averaged spherical-wavefront alpha for each localisation.

    For each localisation at pupil position (u, v), averages the spherical
    alpha over a grid of m×m sub-samples covering the microlens aperture
    (width = uv_scaling).  NaN values (outside the pupil) are ignored.

    Args:
        uv:         (N, 2) array of (u, v) pupil coordinates.
        uv_scaling: Microlens aperture width in pupil units (mla_to_uv_scale).
        na:         Numerical aperture.
        n:          Refractive index of specimen medium.
        m:          Sub-sampling grid size (m+1 points per axis).
        out:        (N, 2) output array — modified in-place.
    """
    ds2 = uv_scaling / 2.0
    na_n = na / n

    for i in range(uv.shape[0]):
        range_u = np.linspace(uv[i, 0] - ds2, uv[i, 0] + ds2, m + 1)
        range_v = np.linspace(uv[i, 1] - ds2, uv[i, 1] + ds2, m + 1)
        um, vm = np.meshgrid(range_u, range_v)

        rho_sq = um ** 2 + vm ** 2
        dr_sq  = 1.0 - rho_sq * na_n ** 2
        dr_sq[dr_sq < 0.0] = np.nan
        phi = -na_n / np.sqrt(dr_sq)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[i, 0] = np.nanmean(um * phi)
            out[i, 1] = np.nanmean(vm * phi)


# ---------------------------------------------------------------------------
# Convenience wrappers (used by main.py)
# ---------------------------------------------------------------------------

def assign_to_lenses(locs_2d_csv: npt.NDArray[float],
                     mla: MicroLensArray,
                     lfm: FourierMicroscope) -> Localisations:
    """Build a Localisations object and assign each point to its nearest lens.

    Args:
        locs_2d_csv: (N, 8) array from preprocessing.center_localizations.
        mla: Rotated MicroLensArray.
        lfm: FourierMicroscope instance.

    Returns:
        Localisations with U, V and lens index populated.
    """
    lfl = Localisations(locs_2d_csv)
    lfl.assign_to_lenses(mla, lfm)
    return lfl


def compute_alpha_model(lfl: Localisations,
                        lfm: FourierMicroscope,
                        model: str = "INTEGRATE_SPHERE",
                        worker_count: int = 1) -> None:
    """Wrapper: compute alpha model in-place on lfl.filtered_locs_2d."""
    lfl.compute_alpha_model(lfm, model=model, worker_count=worker_count)
