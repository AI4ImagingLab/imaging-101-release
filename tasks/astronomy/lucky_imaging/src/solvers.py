"""
Solvers for Lucky Imaging
=========================

Frame ranking, global alignment, alignment point grid construction,
local frame selection, local shift computation, and weighted stacking.
Extracted from PlanetarySystemStacker.

Functions
---------
rank_frames              Rank all frames by global sharpness
find_alignment_rect      Find the best patch for global alignment
align_frames_global      Globally align all frames
create_ap_grid           Create a staggered alignment point grid
rank_frames_local        Rank and select frames per AP
compute_local_shifts     Compute local warp at each AP
stack_and_blend          Stack selected frames and blend into output
one_dim_weight           Compute 1D triangular blending weight ramp
unsharp_mask             Post-processing sharpening via unsharp masking
"""

import math

import cv2
import numpy as np
from scipy import ndimage

from src.physics_model import (
    quality_measure_gradient,
    quality_measure_laplace,
    quality_measure_sobel,
    quality_measure,
    quality_measure_threshold_weighted,
    multilevel_correlation,
)


def rank_frames(frames_data, method="Laplace", normalize=True, stride=2):
    """Rank all frames by sharpness quality.

    Parameters
    ----------
    frames_data : dict -- output of prepare_all_frames()
    method : str -- "Laplace", "Gradient", or "Sobel"
    normalize : bool -- divide quality by brightness
    stride : int -- downsampling stride

    Returns
    -------
    quality_scores : ndarray (N,) -- normalised quality scores (max = 1)
    sorted_indices : ndarray (N,) -- frame indices sorted by descending quality
    """
    n = frames_data['n_frames']
    raw_scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if method == "Laplace":
            q = quality_measure_laplace(frames_data['blurred'][i], stride)
        elif method == "Gradient":
            q = quality_measure_gradient(frames_data['blurred'][i], stride)
        elif method == "Sobel":
            q = quality_measure_sobel(frames_data['blurred'][i], stride)
        else:
            raise ValueError(f"Unknown ranking method: {method}")

        if normalize:
            q /= frames_data['brightness'][i]

        raw_scores[i] = q

    # Normalize so max = 1
    max_score = raw_scores.max()
    if max_score > 0:
        quality_scores = raw_scores / max_score
    else:
        quality_scores = raw_scores

    sorted_indices = np.argsort(-quality_scores)
    return quality_scores, sorted_indices


def find_alignment_rect(reference_frame_blurred, scale_factor=3.0,
                         stride=2, black_threshold=40.0, min_fraction=0.7,
                         search_width=34, border_width=10):
    """Find the best patch for global alignment.

    Parameters
    ----------
    reference_frame_blurred : ndarray, shape (H, W), dtype uint16
    scale_factor : float -- frame dimension / rect dimension
    stride : int
    black_threshold : float
    min_fraction : float
    search_width : int -- alignment search width (rect must be inset by this much)
    border_width : int -- additional border margin

    Returns
    -------
    rect : tuple -- (y_low, y_high, x_low, x_high) of best alignment patch
    """
    h, w = reference_frame_blurred.shape
    rect_h = int(h / scale_factor)
    rect_w = int(w / scale_factor)
    step_y = max(rect_h // 2, 1)
    step_x = max(rect_w // 2, 1)

    # Ensure the alignment rect is far enough from borders so the
    # multilevel_correlation search window fits within the frame.
    margin = search_width + border_width
    y_start = max(margin, 0)
    x_start = max(margin, 0)
    y_end = h - rect_h - margin
    x_end = w - rect_w - margin

    # Fallback: if margins are too large, use the center
    if y_start >= y_end or x_start >= x_end:
        cy, cx = h // 2, w // 2
        return (cy - rect_h // 2, cy + rect_h // 2,
                cx - rect_w // 2, cx + rect_w // 2)

    best_quality = -1.0
    best_rect = (y_start, y_start + rect_h, x_start, x_start + rect_w)

    for y in range(y_start, y_end + 1, step_y):
        for x in range(x_start, x_end + 1, step_x):
            patch = reference_frame_blurred[y:y + rect_h, x:x + rect_w]
            q = quality_measure_threshold_weighted(
                patch, stride, black_threshold, min_fraction
            )
            if q > best_quality:
                best_quality = q
                best_rect = (y, y + rect_h, x, x + rect_w)

    return best_rect


def align_frames_global(frames_data, sorted_indices, rect, search_width=34,
                         average_frame_percent=5):
    """Globally align all frames to a common reference.

    Uses multi-level cross-correlation for alignment.

    Parameters
    ----------
    frames_data : dict -- output of prepare_all_frames()
    sorted_indices : ndarray -- frame indices sorted by quality
    rect : tuple -- (y_low, y_high, x_low, x_high) alignment rectangle
    search_width : int -- maximum search radius
    average_frame_percent : int -- % of best frames for mean reference

    Returns
    -------
    shifts : ndarray (N, 2) -- (dy, dx) shift for each frame
    intersection : tuple -- (y_low, y_high, x_low, x_high) common overlap
    mean_frame : ndarray -- mean of best globally-aligned frames (int32)
    """
    n = frames_data['n_frames']
    h, w = frames_data['shape']
    ry_lo, ry_hi, rx_lo, rx_hi = rect

    shifts = np.zeros((n, 2), dtype=np.int32)
    best_idx = sorted_indices[0]

    # Reference box from best frame
    ref_blurred = frames_data['blurred'][best_idx]
    ref_box = ref_blurred[ry_lo:ry_hi, rx_lo:rx_hi]
    ref_box_phase1 = ref_box[::2, ::2].astype(np.float32)
    ref_box_phase2 = ref_box.astype(np.float32)

    # Align each frame to the reference
    gauss_width = 7  # blur strength for phase 1

    for i in range(n):
        if i == best_idx:
            continue

        frame_blurred = frames_data['blurred'][i]

        shift_y, shift_x, success = multilevel_correlation(
            ref_box_phase1, frame_blurred, gauss_width, ref_box_phase2,
            ry_lo, ry_hi, rx_lo, rx_hi, search_width
        )

        if success:
            shifts[i] = [int(round(shift_y)), int(round(shift_x))]

    # Compute intersection (common overlap of all shifted frames)
    dy_min = shifts[:, 0].min()
    dy_max = shifts[:, 0].max()
    dx_min = shifts[:, 1].min()
    dx_max = shifts[:, 1].max()

    # Intersection region in the coordinate system of the mean frame
    int_y_lo = max(0, -dy_min)
    int_y_hi = min(h, h - dy_max)
    int_x_lo = max(0, -dx_min)
    int_x_hi = min(w, w - dx_max)

    intersection = (int(int_y_lo), int(int_y_hi), int(int_x_lo), int(int_x_hi))
    int_h = int_y_hi - int_y_lo
    int_w = int_x_hi - int_x_lo

    # Compute mean frame from best N% frames
    n_avg = max(int(n * average_frame_percent / 100.0), 1)
    mean_accum = np.zeros((int_h, int_w), dtype=np.float64)

    for k in range(n_avg):
        idx = sorted_indices[k]
        dy, dx = shifts[idx]
        frame_mono = frames_data['blurred'][idx]
        src_y_lo = int_y_lo + dy
        src_y_hi = int_y_hi + dy
        src_x_lo = int_x_lo + dx
        src_x_hi = int_x_hi + dx

        src_y_lo = max(0, int(src_y_lo))
        src_y_hi = min(h, int(src_y_hi))
        src_x_lo = max(0, int(src_x_lo))
        src_x_hi = min(w, int(src_x_hi))

        tgt_h = src_y_hi - src_y_lo
        tgt_w = src_x_hi - src_x_lo
        mean_accum[:tgt_h, :tgt_w] += frame_mono[src_y_lo:src_y_hi,
                                                   src_x_lo:src_x_hi].astype(np.float64)

    mean_frame = (mean_accum / n_avg).astype(np.int32)
    return shifts, intersection, mean_frame


def _ap_locations(num_pixels, min_boundary_distance, step_size, even):
    """Compute 1D alignment point grid locations.

    Parameters
    ----------
    num_pixels : int
    min_boundary_distance : int
    step_size : int
    even : bool

    Returns
    -------
    locations : list of int
    """
    try:
        num_interior_odd = int(math.ceil(
            (num_pixels - 2 * min_boundary_distance) / step_size
        ))
        num_interior_even = num_interior_odd + 1
        distance = (num_pixels - 2 * min_boundary_distance) / num_interior_odd

        if even:
            locations = [int(min_boundary_distance + i * distance)
                         for i in range(num_interior_even)]
        else:
            locations = [int(min_boundary_distance + 0.5 * distance + i * distance)
                         for i in range(num_interior_odd)]
    except Exception:
        locations = []
    return locations


def create_ap_grid(mean_frame, half_box_width=24, structure_threshold=0.04,
                    brightness_threshold=10, search_width=14,
                    dim_fraction_threshold=0.6):
    """Create a staggered alignment point grid.

    Parameters
    ----------
    mean_frame : ndarray, shape (H, W), dtype int32 -- blurred mean frame
    half_box_width : int -- half-width of each AP box
    structure_threshold : float -- minimum normalised structure (0-1)
    brightness_threshold : int -- minimum brightness to keep an AP (8-bit scale)
    search_width : int -- local search radius at each AP
    dim_fraction_threshold : float -- fraction of dim pixels to trigger CoM shift

    Returns
    -------
    alignment_points : list of dict
    """
    half_patch_width = round(half_box_width * 3 / 2)
    step_size = round(half_patch_width * 3 / 2)
    num_y, num_x = mean_frame.shape

    # Brightness threshold in 16-bit scale
    bright_thresh_16 = brightness_threshold * 256

    min_boundary = max(half_box_width + search_width, half_patch_width)

    ap_y = _ap_locations(num_y, min_boundary, step_size, True)
    ap_x_even = _ap_locations(num_x, min_boundary, step_size, True)
    ap_x_odd = _ap_locations(num_x, min_boundary, step_size, False)

    if not ap_y or not ap_x_even or not ap_x_odd:
        return []

    alignment_points = []
    even = True

    for iy, y in enumerate(ap_y):
        extend_y_low = (iy == 0)
        extend_y_high = (iy == len(ap_y) - 1)

        ap_x = ap_x_even if even else ap_x_odd

        for ix, x in enumerate(ap_x):
            extend_x_low = (ix == 0)
            extend_x_high = (ix == len(ap_x) - 1)

            ap = _new_alignment_point(
                y, x, half_box_width, half_patch_width, search_width,
                num_y, num_x, extend_x_low, extend_x_high,
                extend_y_low, extend_y_high, mean_frame
            )
            if ap is None:
                continue

            ref_box = ap['reference_box']
            max_b = float(np.amax(ref_box))
            min_b = float(np.amin(ref_box))

            if max_b > bright_thresh_16 and max_b - min_b > 0:
                # Check dim fraction and potentially shift AP center
                fraction = (ref_box < bright_thresh_16).sum() / ref_box.size
                if fraction > dim_fraction_threshold:
                    com = ndimage.center_of_mass(ref_box.astype(np.float64))
                    y_new = y + int(com[0]) - half_box_width
                    x_new = x + int(com[1]) - half_box_width
                    y_new = max(min_boundary, min(y_new, num_y - min_boundary))
                    x_new = max(min_boundary, min(x_new, num_x - min_boundary))
                    ap = _new_alignment_point(
                        y_new, x_new, half_box_width, half_patch_width,
                        search_width, num_y, num_x,
                        extend_x_low, extend_x_high,
                        extend_y_low, extend_y_high, mean_frame
                    )
                    if ap is None:
                        continue

                ap['structure'] = quality_measure(ap['reference_box'])
                alignment_points.append(ap)

        even = not even

    # Normalize structure and filter
    if alignment_points:
        struct_max = max(ap['structure'] for ap in alignment_points)
        if struct_max > 0:
            for ap in alignment_points:
                ap['structure'] /= struct_max
        alignment_points = [ap for ap in alignment_points
                            if ap['structure'] >= structure_threshold]

    return alignment_points


def _new_alignment_point(y, x, half_box_width, half_patch_width, search_width,
                          num_y, num_x, extend_x_low, extend_x_high,
                          extend_y_low, extend_y_high, mean_frame):
    """Create a single alignment point dict."""
    min_boundary = max(half_box_width + search_width, half_patch_width)
    if (y < min_boundary or y > num_y - min_boundary or
            x < min_boundary or x > num_x - min_boundary):
        return None

    ap = {
        'y': y,
        'x': x,
        'half_box_width': half_box_width,
        'box_y_low': y - half_box_width,
        'box_y_high': y + half_box_width,
        'box_x_low': x - half_box_width,
        'box_x_high': x + half_box_width,
    }

    ap['patch_y_low'] = y - half_patch_width
    ap['patch_y_high'] = y + half_patch_width
    if extend_y_low:
        ap['patch_y_low'] = 0
    elif extend_y_high:
        ap['patch_y_high'] = num_y

    ap['patch_x_low'] = x - half_patch_width
    ap['patch_x_high'] = x + half_patch_width
    if extend_x_low:
        ap['patch_x_low'] = 0
    elif extend_x_high:
        ap['patch_x_high'] = num_x

    ap['reference_box'] = mean_frame[ap['box_y_low']:ap['box_y_high'],
                                      ap['box_x_low']:ap['box_x_high']].copy()

    return ap


def rank_frames_local(frames_data, alignment_points, global_shifts,
                       frame_percent=10, method="Laplace", stride=2,
                       normalize=True):
    """Rank and select frames independently at each alignment point.

    Parameters
    ----------
    frames_data : dict
    alignment_points : list of dict
    global_shifts : ndarray (N, 2)
    frame_percent : int -- percentage of best frames to select per AP
    method : str -- quality metric
    stride : int
    normalize : bool

    Returns
    -------
    alignment_points : list of dict -- updated with 'selected_frames' key
    """
    n = frames_data['n_frames']
    h, w = frames_data['shape']
    stack_size = max(int(math.ceil(n * frame_percent / 100.0)), 1)

    for ap in alignment_points:
        qualities = []
        for i in range(n):
            dy, dx = global_shifts[i]
            y_lo = max(0, ap['patch_y_low'] + dy)
            y_hi = min(h, ap['patch_y_high'] + dy)
            x_lo = max(0, ap['patch_x_low'] + dx)
            x_hi = min(w, ap['patch_x_high'] + dx)

            frame = frames_data['blurred'][i]
            patch = frame[y_lo:y_hi, x_lo:x_hi]

            if method == "Laplace":
                q = float(cv2.meanStdDev(
                    cv2.Laplacian(patch[::stride, ::stride], cv2.CV_32F)
                )[1][0][0])
            elif method == "Gradient":
                q = quality_measure_gradient(patch, stride)
            else:
                q = quality_measure_sobel(patch, stride)

            if normalize:
                q /= frames_data['brightness'][i]

            qualities.append(q)

        # Select top stack_size frames
        sorted_idx = sorted(range(n), key=lambda k: qualities[k], reverse=True)
        ap['selected_frames'] = sorted_idx[:stack_size]
        ap['stack_size'] = stack_size

    return alignment_points


def compute_local_shifts(frames_data, alignment_points, global_shifts,
                          search_width=14, gauss_width=7):
    """Compute local warp shifts at each alignment point.

    Parameters
    ----------
    frames_data : dict
    alignment_points : list of dict -- with 'selected_frames'
    global_shifts : ndarray (N, 2)
    search_width : int
    gauss_width : int

    Returns
    -------
    alignment_points : list of dict -- updated with 'local_shifts' key
    """
    # Prepare reference boxes for multilevel correlation
    for ap in alignment_points:
        ref_box = ap['reference_box'].astype(np.float32)
        ap['ref_box_phase1'] = ref_box[::2, ::2]
        ap['ref_box_phase2'] = ref_box

    # Compute penalty weight matrix for first phase
    sw2 = 4
    sw1 = int((search_width - sw2) / 2)
    extent = 2 * sw1 + 1
    penalty_factor = 0.00025
    weight_matrix = np.empty((extent, extent), dtype=np.float32)
    for y in range(extent):
        for x in range(extent):
            weight_matrix[y, x] = 1.0 - penalty_factor * (
                (y / sw1 - 1) ** 2 + (x / sw1 - 1) ** 2
            )

    for ap in alignment_points:
        local_shifts = {}
        for frame_idx in ap['selected_frames']:
            dy, dx = global_shifts[frame_idx]
            frame_blurred = frames_data['blurred'][frame_idx]

            y_lo = ap['box_y_low'] + dy
            y_hi = ap['box_y_high'] + dy
            x_lo = ap['box_x_low'] + dx
            x_hi = ap['box_x_high'] + dx

            shift_y, shift_x, success = multilevel_correlation(
                ap['ref_box_phase1'], frame_blurred, gauss_width,
                ap['ref_box_phase2'],
                y_lo, y_hi, x_lo, x_hi, search_width,
                weight_matrix_first_phase=weight_matrix,
                subpixel_solve=False
            )

            if success:
                local_shifts[frame_idx] = (int(round(shift_y)), int(round(shift_x)))
            else:
                local_shifts[frame_idx] = (0, 0)

        ap['local_shifts'] = local_shifts

    return alignment_points


def one_dim_weight(patch_low, patch_high, box_center,
                    extend_low=False, extend_high=False):
    """Compute 1D triangular blending weight ramp.

    Parameters
    ----------
    patch_low : int -- lower patch index
    patch_high : int -- upper patch index
    box_center : int -- AP centre coordinate
    extend_low : bool -- set lower ramp to 1.0
    extend_high : bool -- set upper ramp to 1.0

    Returns
    -------
    weights : ndarray (patch_high - patch_low,), dtype float32
    """
    patch_size = patch_high - patch_low
    center_offset = box_center - patch_low
    weights = np.empty(patch_size, dtype=np.float32)

    if extend_low:
        weights[:center_offset] = 1.0
    else:
        weights[:center_offset] = np.arange(1, center_offset + 1, dtype=np.float32) / \
                                   np.float32(center_offset + 1)

    upper_len = patch_size - center_offset
    if extend_high:
        weights[center_offset:] = 1.0
    else:
        weights[center_offset:] = np.arange(upper_len, 0, -1, dtype=np.float32) / \
                                   np.float32(upper_len)

    return weights


def stack_and_blend(frames, alignment_points, global_shifts, intersection,
                     mean_frame, drizzle_factor=1, normalize_brightness=True):
    """Stack selected frames per AP and blend into final image.

    Parameters
    ----------
    frames : ndarray (N, H, W, 3), dtype uint8 -- original RGB frames
    alignment_points : list of dict -- with selected_frames and local_shifts
    global_shifts : ndarray (N, 2)
    intersection : tuple -- (y_low, y_high, x_low, x_high)
    mean_frame : ndarray -- globally-aligned mean for background fill
    drizzle_factor : int -- super-resolution factor (1 = off)
    normalize_brightness : bool -- normalise frame brightness

    Returns
    -------
    stacked : ndarray, shape (H', W', 3), dtype uint16 -- 16-bit stacked image
    """
    int_y_lo, int_y_hi, int_x_lo, int_x_hi = intersection
    dim_y = int_y_hi - int_y_lo
    dim_x = int_x_hi - int_x_lo
    dim_y_d = dim_y * drizzle_factor
    dim_x_d = dim_x * drizzle_factor
    n_frames = frames.shape[0]
    color = frames.ndim == 4 and frames.shape[3] == 3

    # Allocate stacking buffer and weight accumulator
    if color:
        stacked_buffer = np.zeros((dim_y_d, dim_x_d, 3), dtype=np.float32)
    else:
        stacked_buffer = np.zeros((dim_y_d, dim_x_d), dtype=np.float32)
    weight_sum = np.full((dim_y_d, dim_x_d), 1e-30, dtype=np.float32)

    # Compute brightness normalization factor
    if normalize_brightness:
        from statistics import median as stat_median
        brightnesses = []
        for i in range(n_frames):
            mono = cv2.cvtColor(
                cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY
            )
            brightnesses.append(float(cv2.mean(mono)[0]) + 1e-7)
        median_bright = stat_median(brightnesses)
    else:
        brightnesses = [1.0] * n_frames
        median_bright = 1.0

    # Initialize AP stacking buffers and compute drizzled coordinates
    for ap in alignment_points:
        ap['patch_y_low_d'] = (ap['patch_y_low'] - int_y_lo) * drizzle_factor
        ap['patch_y_high_d'] = (ap['patch_y_high'] - int_y_lo) * drizzle_factor
        ap['patch_x_low_d'] = (ap['patch_x_low'] - int_x_lo) * drizzle_factor
        ap['patch_x_high_d'] = (ap['patch_x_high'] - int_x_lo) * drizzle_factor
        ap['y_d'] = (ap['y'] - int_y_lo) * drizzle_factor
        ap['x_d'] = (ap['x'] - int_x_lo) * drizzle_factor

        # Clip to output bounds
        ap['patch_y_low_d'] = max(0, ap['patch_y_low_d'])
        ap['patch_y_high_d'] = min(dim_y_d, ap['patch_y_high_d'])
        ap['patch_x_low_d'] = max(0, ap['patch_x_low_d'])
        ap['patch_x_high_d'] = min(dim_x_d, ap['patch_x_high_d'])
        ap['y_d'] = max(0, min(ap['y_d'], dim_y_d - 1))
        ap['x_d'] = max(0, min(ap['x_d'], dim_x_d - 1))

        buf_h = ap['patch_y_high_d'] - ap['patch_y_low_d']
        buf_w = ap['patch_x_high_d'] - ap['patch_x_low_d']
        if color:
            ap['stacking_buffer'] = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
        else:
            ap['stacking_buffer'] = np.zeros((buf_h, buf_w), dtype=np.float32)

    # Compute blending weights for each AP
    stack_size = alignment_points[0].get('stack_size', 10) if alignment_points else 10

    for ap in alignment_points:
        py_lo = ap['patch_y_low_d']
        py_hi = ap['patch_y_high_d']
        px_lo = ap['patch_x_low_d']
        px_hi = ap['patch_x_high_d']

        extend_low_y = (py_lo == 0)
        extend_high_y = (py_hi == dim_y_d)
        extend_low_x = (px_lo == 0)
        extend_high_x = (px_hi == dim_x_d)

        wy = one_dim_weight(py_lo, py_hi, ap['y_d'],
                             extend_low=extend_low_y, extend_high=extend_high_y)
        wx = one_dim_weight(px_lo, px_hi, ap['x_d'],
                             extend_low=extend_low_x, extend_high=extend_high_x)
        ap['weights_yx'] = np.minimum(wy[:, np.newaxis], wx[np.newaxis, :])

        weight_sum[py_lo:py_hi, px_lo:px_hi] += float(stack_size) * ap['weights_yx']

    # Check for stacking holes (regions not covered by any AP)
    has_holes = np.count_nonzero(weight_sum < 1e-10) > 0

    # Stack frames
    h_frame, w_frame = frames.shape[1], frames.shape[2]
    border_y_lo = border_y_hi = border_x_lo = border_x_hi = 0

    # Build frame-to-AP mapping for efficiency
    frame_to_aps = {i: [] for i in range(n_frames)}
    for ap_idx, ap in enumerate(alignment_points):
        for fi in ap['selected_frames']:
            frame_to_aps[fi].append(ap_idx)

    # Background accumulator
    if has_holes:
        if color:
            bg_accum = np.zeros((dim_y, dim_x, 3), dtype=np.float64)
        else:
            bg_accum = np.zeros((dim_y, dim_x), dtype=np.float64)
        bg_count = 0

    for fi in range(n_frames):
        if not frame_to_aps[fi] and not has_holes:
            continue

        # Brightness-normalized frame as float32
        bright_scale = median_bright / brightnesses[fi] if normalize_brightness else 1.0
        frame_f = frames[fi].astype(np.float32) * bright_scale

        dy, dx = int(global_shifts[fi, 0]), int(global_shifts[fi, 1])

        # Stack into AP buffers
        for ap_idx in frame_to_aps[fi]:
            ap = alignment_points[ap_idx]
            local_dy, local_dx = ap['local_shifts'].get(fi, (0, 0))

            total_dy = dy - local_dy
            total_dx = dx - local_dx

            # Source region in the frame
            py_lo = ap['patch_y_low'] + int_y_lo
            py_hi = ap['patch_y_high'] + int_y_lo
            px_lo = ap['patch_x_low'] + int_x_lo
            px_hi = ap['patch_x_high'] + int_x_lo

            buf_h = ap['patch_y_high_d'] - ap['patch_y_low_d']
            buf_w = ap['patch_x_high_d'] - ap['patch_x_low_d']

            # Source coordinates with shift
            src_y_lo = py_lo + total_dy
            src_y_hi = py_hi + total_dy
            src_x_lo = px_lo + total_dx
            src_x_hi = px_hi + total_dx

            # Clip to frame bounds
            tgt_y_lo = 0
            tgt_x_lo = 0
            if src_y_lo < 0:
                tgt_y_lo = -src_y_lo
                border_y_lo = max(border_y_lo, tgt_y_lo)
                src_y_lo = 0
            if src_y_hi > h_frame:
                border_y_hi = max(border_y_hi, src_y_hi - h_frame)
                src_y_hi = h_frame
            if src_x_lo < 0:
                tgt_x_lo = -src_x_lo
                border_x_lo = max(border_x_lo, tgt_x_lo)
                src_x_lo = 0
            if src_x_hi > w_frame:
                border_x_hi = max(border_x_hi, src_x_hi - w_frame)
                src_x_hi = w_frame

            tgt_y_hi = tgt_y_lo + (src_y_hi - src_y_lo)
            tgt_x_hi = tgt_x_lo + (src_x_hi - src_x_lo)

            if tgt_y_hi <= tgt_y_lo or tgt_x_hi <= tgt_x_lo:
                continue
            if tgt_y_hi > buf_h or tgt_x_hi > buf_w:
                tgt_y_hi = min(tgt_y_hi, buf_h)
                tgt_x_hi = min(tgt_x_hi, buf_w)
                src_y_hi = src_y_lo + (tgt_y_hi - tgt_y_lo)
                src_x_hi = src_x_lo + (tgt_x_hi - tgt_x_lo)

            ap['stacking_buffer'][tgt_y_lo:tgt_y_hi, tgt_x_lo:tgt_x_hi] += \
                frame_f[src_y_lo:src_y_hi, src_x_lo:src_x_hi]

        # Background contribution
        if has_holes:
            src_y = int_y_lo + dy
            src_x = int_x_lo + dx
            if 0 <= src_y and src_y + dim_y <= h_frame and \
               0 <= src_x and src_x + dim_x <= w_frame:
                bg_accum += frame_f[src_y:src_y + dim_y,
                                     src_x:src_x + dim_x].astype(np.float64)
                bg_count += 1

    # Merge AP buffers
    for ap in alignment_points:
        py_lo = ap['patch_y_low_d']
        py_hi = ap['patch_y_high_d']
        px_lo = ap['patch_x_low_d']
        px_hi = ap['patch_x_high_d']

        if color:
            stacked_buffer[py_lo:py_hi, px_lo:px_hi, :] += \
                ap['stacking_buffer'] * ap['weights_yx'][:, :, np.newaxis]
        else:
            stacked_buffer[py_lo:py_hi, px_lo:px_hi] += \
                ap['stacking_buffer'] * ap['weights_yx']

    # Normalize by weight sum
    if color:
        stacked_buffer /= weight_sum[:, :, np.newaxis]
    else:
        stacked_buffer /= weight_sum

    # Blend with background if needed
    if has_holes and bg_count > 0:
        bg_avg = (bg_accum / bg_count).astype(np.float32)
        bg_threshold = 0.2
        fg_weight = weight_sum / (bg_threshold * stack_size)
        np.clip(fg_weight, 0.0, 1.0, out=fg_weight)
        if color:
            stacked_buffer = ((stacked_buffer - bg_avg) *
                              fg_weight[:, :, np.newaxis] + bg_avg)
        else:
            stacked_buffer = (stacked_buffer - bg_avg) * fg_weight + bg_avg

    # Trim borders
    if border_y_lo or border_y_hi or border_x_lo or border_x_hi:
        y_end = dim_y_d - border_y_hi if border_y_hi else dim_y_d
        x_end = dim_x_d - border_x_hi if border_x_hi else dim_x_d
        stacked_buffer = stacked_buffer[border_y_lo:y_end, border_x_lo:x_end]

    # Convert to uint16
    stacked_buffer = np.clip(stacked_buffer / 255.0, 0.0, 1.0)
    stacked = (stacked_buffer * 65535).astype(np.uint16)

    return stacked


def unsharp_mask(image, sigma=2.0, alpha=1.5):
    """Post-processing sharpening via unsharp masking.

    Stacking reduces noise but also slightly softens the result due to
    residual sub-pixel misalignment. Unsharp masking recovers this lost
    sharpness, leveraging the improved SNR from stacking (sharpening a
    single noisy frame would amplify noise, but sharpening the low-noise
    stack produces a clean, sharp result).

    sharpened = original + alpha * (original - GaussianBlur(original, sigma))

    Parameters
    ----------
    image : ndarray, dtype uint16 -- stacked image
    sigma : float -- Gaussian blur sigma for the low-pass component
    alpha : float -- sharpening strength (1.0 = moderate, 2.0 = strong)

    Returns
    -------
    sharpened : ndarray, dtype uint16 -- sharpened image
    """
    img_f = image.astype(np.float32)
    ksize = int(sigma * 6) | 1  # ensure odd kernel size
    blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
    sharpened = img_f + alpha * (img_f - blurred)
    return np.clip(sharpened, 0, 65535).astype(np.uint16)
