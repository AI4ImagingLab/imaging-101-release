"""
Physics model for Lucky Imaging
================================

Quality metrics for measuring atmospheric seeing, and image registration
algorithms for aligning frames. Extracted from PlanetarySystemStacker.

Functions
---------
quality_measure_gradient     Frame quality via finite-difference gradient
quality_measure_laplace      Frame quality via Laplacian std deviation
quality_measure_sobel        Frame quality via Sobel gradient energy
quality_measure              Structure measure for alignment point filtering
multilevel_correlation       Two-phase normalised cross-correlation alignment
sub_pixel_solve              2D paraboloid fit for sub-pixel refinement
phase_correlation            FFT-based global shift detection
"""

import cv2
import numpy as np
from numpy import diff, average, hypot, matmul
from numpy.fft import fft2, ifft2
from scipy.ndimage import sobel


def quality_measure_gradient(frame, stride=2):
    """Compute frame quality via finite-difference gradient magnitude.

    Parameters
    ----------
    frame : ndarray, shape (H, W) -- monochrome blurred frame (uint16)
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- average gradient magnitude
    """
    frame_strided = frame[::stride, ::stride]
    dx = diff(frame_strided)[1:, :]
    dy = diff(frame_strided, axis=0)[:, 1:]
    dnorm = hypot(dx.astype(np.float64), dy.astype(np.float64))
    return float(average(dnorm))


def quality_measure_laplace(frame_blurred, stride=2):
    """Compute frame quality via Laplacian standard deviation.

    Parameters
    ----------
    frame_blurred : ndarray, shape (H, W), dtype uint16 -- Gaussian-blurred frame
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- std of Laplacian (higher = sharper)
    """
    subsampled = frame_blurred[::stride, ::stride]
    lap = cv2.Laplacian(subsampled, cv2.CV_32F)
    return float(cv2.meanStdDev(lap)[1][0][0])


def quality_measure_sobel(frame, stride=2):
    """Compute frame quality via Sobel gradient energy.

    Parameters
    ----------
    frame : ndarray, shape (H, W) -- monochrome blurred frame
    stride : int -- downsampling stride

    Returns
    -------
    quality : float -- sum of Sobel gradient magnitudes
    """
    frame_int32 = frame[::stride, ::stride].astype('int32')
    dx = sobel(frame_int32, 0)
    dy = sobel(frame_int32, 1)
    mag = hypot(dx, dy)
    return float(mag.sum())


def quality_measure(frame):
    """Measure structure in a rectangular frame via finite-difference gradient.

    Used for filtering alignment points by structure content.

    Parameters
    ----------
    frame : ndarray, shape (H, W) -- image patch

    Returns
    -------
    quality : float -- min of avg |dx|, avg |dy|
    """
    dx = diff(frame.astype(np.float64))
    dy = diff(frame.astype(np.float64), axis=0)
    sharpness_x = average(np.abs(dx))
    sharpness_y = average(np.abs(dy))
    return float(min(sharpness_x, sharpness_y))


def quality_measure_threshold_weighted(frame, stride=2, black_threshold=40.0,
                                        min_fraction=0.7):
    """Measure structure with brightness threshold weighting.

    Used for scoring alignment rectangle candidates.

    Parameters
    ----------
    frame : ndarray, shape (H, W) -- image patch
    stride : int
    black_threshold : float
    min_fraction : float

    Returns
    -------
    quality : float
    """
    frame_size = frame.shape[0] * frame.shape[1]
    stride_2 = 2 * stride
    frame_f = frame.astype(np.float64)

    mask = frame_f > black_threshold
    mask_fraction = mask.sum() / frame_size

    if mask_fraction > min_fraction:
        sum_h = np.abs(
            (frame_f[:, stride_2:] - frame_f[:, :-stride_2])[mask[:, stride:-stride]]
        ).sum() / mask_fraction
        sum_v = np.abs(
            (frame_f[stride_2:, :] - frame_f[:-stride_2, :])[mask[stride:-stride, :]]
        ).sum() / mask_fraction
    else:
        sum_h = np.abs(
            (frame_f[:, stride_2:] - frame_f[:, :-stride_2])[mask[:, stride:-stride]]
        ).sum()
        sum_v = np.abs(
            (frame_f[stride_2:, :] - frame_f[:-stride_2, :])[mask[stride:-stride, :]]
        ).sum()

    return float(min(sum_h, sum_v))


# Precomputed matrix for sub_pixel_solve: inv(A^T A) * A^T for the 2D
# paraboloid least-squares fit on a 3x3 grid.
_SUB_PIXEL_SOLVE_MATRIX = np.array([
    [0.16666667, -0.33333333, 0.16666667, 0.16666667, -0.33333333,
     0.16666667, 0.16666667, -0.33333333, 0.16666667],
    [0.16666667, 0.16666667, 0.16666667, -0.33333333, -0.33333333,
     -0.33333333, 0.16666667, 0.16666667, 0.16666667],
    [0.25, 0., -0.25, 0., 0., 0., -0.25, 0., 0.25],
    [-0.16666667, 0., 0.16666667, -0.16666667, 0., 0.16666667,
     -0.16666667, 0., 0.16666667],
    [-0.16666667, -0.16666667, -0.16666667, 0., 0., 0.,
     0.16666667, 0.16666667, 0.16666667],
    [-0.11111111, 0.22222222, -0.11111111, 0.22222222, 0.55555556,
     0.22222222, -0.11111111, 0.22222222, -0.11111111],
])


def sub_pixel_solve(function_values):
    """Fit a 2D paraboloid to a 3x3 grid and find the sub-pixel extremum.

    f(dy, dx) = a*dy^2 + b*dx^2 + c*dy*dx + d*dy + e*dx + g

    Parameters
    ----------
    function_values : ndarray, shape (3, 3) -- correlation or deviation values

    Returns
    -------
    dy : float -- sub-pixel vertical correction
    dx : float -- sub-pixel horizontal correction
    """
    vals = function_values.reshape(9).astype(np.float64)
    a_f, b_f, c_f, d_f, e_f, g_f = matmul(_SUB_PIXEL_SOLVE_MATRIX, vals)

    denominator = c_f ** 2 - 4.0 * a_f * b_f
    if abs(denominator) > 1e-10 and abs(a_f) > 1e-10:
        y_correction = (2.0 * a_f * e_f - c_f * d_f) / denominator
        x_correction = (-c_f * y_correction - d_f) / (2.0 * a_f)
    elif abs(denominator) > 1e-10 and abs(c_f) > 1e-10:
        y_correction = (2.0 * a_f * e_f - c_f * d_f) / denominator
        x_correction = (-2.0 * b_f * y_correction - e_f) / c_f
    else:
        y_correction = 0.0
        x_correction = 0.0

    return y_correction, x_correction


def multilevel_correlation(reference_box_first_phase, frame_mono_blurred,
                           blur_strength, reference_box_second_phase,
                           y_low, y_high, x_low, x_high, search_width,
                           weight_matrix_first_phase=None, subpixel_solve=False):
    """Two-phase normalised cross-correlation alignment.

    Phase 1: coarsened (stride 2) search over the full search window.
    Phase 2: full-resolution refinement within +/-4 pixels of phase-1 result.
    Optional sub-pixel paraboloid fit.

    Parameters
    ----------
    reference_box_first_phase : ndarray -- stride-2 reference patch (float32)
    frame_mono_blurred : ndarray -- full blurred mono frame (uint16)
    blur_strength : int -- additional Gaussian blur kernel size for phase 1
    reference_box_second_phase : ndarray -- full-res reference patch (float32)
    y_low, y_high, x_low, x_high : int -- box bounds in frame coords
    search_width : int -- max shift in pixels
    weight_matrix_first_phase : ndarray or None -- penalty weighting
    subpixel_solve : bool -- enable sub-pixel refinement

    Returns
    -------
    shift_y : float -- total vertical shift
    shift_x : float -- total horizontal shift
    success : bool -- True if correlation peak is valid
    """
    search_width_second_phase = 4
    search_width_first_phase = int((search_width - search_width_second_phase) / 2)

    # Phase 1: coarse grid
    index_extension = search_width_first_phase * 2
    y_lo_1 = max(0, y_low - index_extension)
    y_hi_1 = min(frame_mono_blurred.shape[0], y_high + index_extension)
    x_lo_1 = max(0, x_low - index_extension)
    x_hi_1 = min(frame_mono_blurred.shape[1], x_high + index_extension)

    frame_window = cv2.GaussianBlur(
        frame_mono_blurred[y_lo_1:y_hi_1:2, x_lo_1:x_hi_1:2],
        (blur_strength, blur_strength), 0
    )

    result = cv2.matchTemplate(
        frame_window.astype(np.float32),
        reference_box_first_phase.astype(np.float32),
        cv2.TM_CCORR_NORMED
    )

    if weight_matrix_first_phase is not None:
        # Ensure weight matrix matches result size
        rh, rw = result.shape[:2]
        wh, ww = weight_matrix_first_phase.shape[:2]
        if rh == wh and rw == ww:
            result_weighted = result * weight_matrix_first_phase
        else:
            result_weighted = result
        _, _, _, max_loc = cv2.minMaxLoc(result_weighted)
    else:
        _, _, _, max_loc = cv2.minMaxLoc(result)

    shift_y_1 = (search_width_first_phase - max_loc[1]) * 2
    shift_x_1 = (search_width_first_phase - max_loc[0]) * 2

    success_1 = (abs(shift_y_1) != index_extension and
                 abs(shift_x_1) != index_extension)

    if not success_1:
        return 0, 0, False

    # Phase 2: fine grid
    y_lo_2 = y_low - shift_y_1 - search_width_second_phase
    y_hi_2 = y_high - shift_y_1 + search_width_second_phase
    x_lo_2 = x_low - shift_x_1 - search_width_second_phase
    x_hi_2 = x_high - shift_x_1 + search_width_second_phase

    # Clip to frame bounds
    y_lo_2 = max(0, y_lo_2)
    y_hi_2 = min(frame_mono_blurred.shape[0], y_hi_2)
    x_lo_2 = max(0, x_lo_2)
    x_hi_2 = min(frame_mono_blurred.shape[1], x_hi_2)

    frame_window_2 = frame_mono_blurred[y_lo_2:y_hi_2, x_lo_2:x_hi_2]

    result_2 = cv2.matchTemplate(
        frame_window_2.astype(np.float32),
        reference_box_second_phase,
        cv2.TM_CCORR_NORMED
    )

    _, _, _, max_loc_2 = cv2.minMaxLoc(result_2)
    shift_y_2 = search_width_second_phase - max_loc_2[1]
    shift_x_2 = search_width_second_phase - max_loc_2[0]

    success_2 = (abs(shift_y_2) != search_width_second_phase and
                 abs(shift_x_2) != search_width_second_phase)

    if not success_2:
        shift_y_2 = 0
        shift_x_2 = 0
    elif subpixel_solve:
        try:
            surroundings = result_2[max_loc_2[1]-1:max_loc_2[1]+2,
                                     max_loc_2[0]-1:max_loc_2[0]+2]
            if surroundings.shape == (3, 3):
                y_corr, x_corr = sub_pixel_solve(surroundings)
                if abs(y_corr) <= 1.0 and abs(x_corr) <= 1.0:
                    shift_y_2 -= y_corr
                    shift_x_2 -= x_corr
        except Exception:
            pass

    return (shift_y_1 + shift_y_2, shift_x_1 + shift_x_2, success_2)


def phase_correlation(frame_0, frame_1, shape):
    """Compute global shift via FFT phase correlation.

    Parameters
    ----------
    frame_0 : ndarray, shape (H, W) -- reference frame
    frame_1 : ndarray, shape (H, W) -- shifted frame
    shape : tuple -- (H, W)

    Returns
    -------
    shift_y : int -- vertical shift
    shift_x : int -- horizontal shift
    """
    f0 = fft2(frame_0.astype(np.float64))
    f1 = fft2(frame_1.astype(np.float64))

    cross = (f0 * f1.conjugate()) / (np.abs(f0) * np.abs(f1) + 1e-30)
    ir = np.abs(ifft2(cross))

    ty, tx = np.unravel_index(np.argmax(ir), shape)

    if ty > shape[0] // 2:
        ty -= shape[0]
    if tx > shape[1] // 2:
        tx -= shape[1]

    return int(ty), int(tx)
