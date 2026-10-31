"""Calibrated stereo geometry and explicit RGB Census acquisition costs.

Census extraction adapted from the AGPL-3.0 MGM reference implementation.
Copyright (C) 2015 Gabriele Facciolo, Carlo de Franchis, Enric Meinhardt.
See the task LICENSE and private validation provenance.
"""
import numpy as np

FOCAL_LENGTH = 1038.018
BASELINE_MM = 176.252
DISPARITY_OFFSET = 53.271
LEFT_CENTRE = (322.037, 243.393)
RIGHT_CENTRE = (375.308, 243.393)

def project_points(points):
    """Project (...,3) left-camera XYZ coordinates in mm into two pixel planes."""
    points = np.asarray(points, dtype=np.float64)
    left = points[..., :2] / points[..., 2, None] * FOCAL_LENGTH + LEFT_CENTRE
    shifted = points[..., :2] - (BASELINE_MM, 0.)
    right = shifted / points[..., 2, None] * FOCAL_LENGTH + RIGHT_CENTRE
    return left, right

def depth_from_disparity(disparity):
    """Convert positive left-minus-right pixel disparity into optical-axis mm."""
    return FOCAL_LENGTH * BASELINE_MM / (np.asarray(disparity, dtype=np.float64) + DISPARITY_OFFSET)

def census_bytes(image):
    """Return HWC-byte descriptors: 72 strict comparisons and 24 zero padding bits.

    RGB channels precede row/column offsets; unavailable neighbours compare false.
    Padding mirrors the original three-float-word descriptor representation.
    """
    image = np.asarray(image, dtype=np.float32)
    h, w, channels = image.shape
    if channels != 3:
        raise ValueError('RGB images required')
    padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), constant_values=np.nan)
    bits = []
    for channel in range(3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx or dy:
                    bits.append(image[..., channel] < padded[2+dy:2+dy+h, 2+dx:2+dx+w, channel])
    packed = np.packbits(np.stack(bits, axis=-1), axis=-1, bitorder='big')
    return np.pad(packed, ((0, 0), (0, 0), (0, 3)))

def cost_volume(left, right, dmin=-127, dmax=0):
    """Mean-per-word Hamming costs for native labels d = u_right - u_left.

    Invalid correspondences cost infinity. If an entire pixel has no candidate,
    the original implementation's zero-cost fallback is retained explicitly.
    """
    a, b = census_bytes(left), census_bytes(right)
    h, w = a.shape[:2]
    result = np.full((h, w, dmax-dmin+1), np.inf, dtype=np.float32)
    popcount = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    for index, disparity in enumerate(range(dmin, dmax+1)):
        lo, hi = max(0, -disparity), min(w, w-disparity)
        if hi > lo:
            xor = a[:, lo:hi] ^ b[:, lo+disparity:hi+disparity]
            # Source computeC_census_on_preprocessed_images returns r*1.0/u.nch;
            # the packed descriptor has three float words, including padding.
            result[:, lo:hi, index] = popcount[xor].sum(axis=-1).astype(np.float64)/3.
    result[~np.isfinite(result).any(axis=-1)] = 0
    return result
