"""
Image loading, preprocessing, and shapelet decomposition of input galaxy images.
"""

import numpy as np
import scipy.ndimage
from PIL import Image

from .physics_model import (
    make_grid, image2array, array2image, re_size,
    shapelet_decomposition, shapelet_function,
)


def load_and_prepare_galaxy(image_path, smooth_sigma=5, downsample_factor=25):
    """Load galaxy image, subtract background, pad to square, smooth, downsample.

    :param image_path: path to JPEG/PNG galaxy image
    :param smooth_sigma: Gaussian smoothing sigma in pixels
    :param downsample_factor: integer factor to reduce resolution
    :return: (ngc_square, ngc_conv, ngc_resized, numPix_resized)
    """
    ngc_data = np.array(Image.open(image_path).convert('F'))
    median = np.median(ngc_data[:200, :200])
    ngc_data -= median

    nx, ny = ngc_data.shape
    n_sq = max(nx, ny)
    ngc_square = np.zeros((n_sq, n_sq))
    x_start = (n_sq - nx) // 2
    y_start = (n_sq - ny) // 2
    ngc_square[x_start:x_start + nx, y_start:y_start + ny] = ngc_data

    ngc_conv = scipy.ndimage.gaussian_filter(ngc_square, smooth_sigma,
                                              mode='nearest', truncate=6)

    numPix_large = len(ngc_conv) // downsample_factor
    n_new = (numPix_large - 1) * downsample_factor
    ngc_cut = ngc_conv[:n_new, :n_new]
    ngc_resized = re_size(ngc_cut, downsample_factor)
    numPix_resized = numPix_large - 1

    return ngc_square, ngc_conv, ngc_resized, numPix_resized


def decompose_shapelets(image_2d, n_max, beta, deltaPix=1.0):
    """Decompose a 2D image into shapelet coefficients.

    :param image_2d: 2D image array (square)
    :param n_max: maximum polynomial order
    :param beta: shapelet scale parameter
    :param deltaPix: pixel size
    :return: coefficient array of length (n_max+1)*(n_max+2)/2
    """
    numPix = image_2d.shape[0]
    x, y = make_grid(numPix, deltapix=deltaPix)
    image_1d = image2array(image_2d)
    return shapelet_decomposition(image_1d, x, y, n_max, beta, deltaPix)


def reconstruct_from_shapelets(coeff, n_max, beta, numPix, deltaPix=1.0):
    """Reconstruct 2D image from shapelet coefficients.

    :return: 2D image array
    """
    x, y = make_grid(numPix, deltapix=deltaPix)
    flux = shapelet_function(x, y, coeff, n_max, beta)
    return array2image(flux)
