"""Explicit Crocker–Grier finding and two-pass moment localization.
Extracted/adapted from sdt-python c9d1c09641e7d781a39a36e6fc89e429bf191ca9.
Copyright2020 Lukas Schrangl. BSD-3-Clause; see LICENSE.
"""
import collections,math
from typing import Tuple,Union
import numpy as np
import scipy.ndimage
from scipy import ndimage
from .preprocessing import bandpass_filter
peak_params=['x','y','mass','size','ecc']
ColumnNums=collections.namedtuple('ColumnNums',peak_params)
col_nums=ColumnNums(**{k:v for v,k in enumerate(peak_params)})

def find_peaks_yx(image, search_radius, threshold):
    """Find local maxima in image

    Finds the locations of all the local maxima in an image with
    intensity greater than threshold.

    The actual finding and filtering function. Usually one would not call
    it directly, but use :py:func:`find`.

    Parameters
    ----------
    image : numpy.ndarray
        The image to analyze
    search_radius : int
        Search for local maxima within this radius. That is, if two local
        maxima are within ``search_radius`` of each other, only the greater
        one will be taken.
    threshold : float
        Minumum peak intensity

    Returns
    -------
    maxima : numpy.ndarray
        Indices of the detected maxima. Each row is a set of indices giving
        where in the `image` array one can find a maximum.
    """
    # create circular mask with radius `search_radius`
    x_sq = np.arange(-search_radius, search_radius + 1)**2
    mask = x_sq[:, np.newaxis] + x_sq[np.newaxis, :]
    mask = (mask <= search_radius**2)

    # use mask to dilate the image
    dil = ndimage.grey_dilation(image, footprint=mask)

    # if threshold is None, determine it from the data
    if threshold is None:
        threshold = np.percentile(image - image.min(), 70) + 1

    # wherever the image value is equal to the dilated value, we have a
    # candidate for a maximum
    candidates = np.nonzero(np.isclose(dil, image) & (image >= threshold))
    candidates = np.transpose(candidates)

    # discard maxima within `search_radius` pixels of the edges
    in_margin = np.any(
        (candidates < search_radius) |
        (candidates >= np.array(image.shape) - search_radius), axis=1)
    candidates = candidates[~in_margin]

    # remove spurious maxima which result from flat peaks
    is_max = np.zeros(len(candidates), dtype=bool)
    candidate_list = tuple(candidates.T)
    max_img = np.zeros(image.shape)
    max_img[candidate_list] = image[candidate_list]

    for cnt, (i, j) in enumerate(candidates):
        roi = max_img[i-search_radius:i+search_radius+1,
                      j-search_radius:j+search_radius+1] * mask

        # complicated, but that's the way the original implementation does it
        # and we want to be compatible
        sort_val = np.sort(roi, axis=0)
        sort_idx = np.argsort(roi, axis=0)
        max_j = np.argmax(sort_val[-1, :])  # column of the max value
        max_i = sort_idx[-1, max_j]         # row of the max value
        if (max_i == search_radius) and (max_j == search_radius):
            # the current value is the maximum in the ROI, since
            # max_i == max_j == search_radius means that the max is in the
            # center of the ROI
            is_max[cnt] = True
        else:
            # delete from max_img
            max_img[i, j] = 0

    return candidates[is_max]


def _get_crop(img: np.ndarray, start_idx: Tuple[int, int],
              end_idx: Tuple[int, int]) -> Union[np.ndarray, None]:
    """Get a crop of an image (i.e., 2d array)

    If the crop is not possible because some index is out of bounds, return
    `None`

    Parameters
    ----------
    img
        Image to crop
    start_idx
        Lower bound indices of the crop
    end_idx
        Upper bound indices of the crop

    Returns
    -------
    Cropped region if possible, else `None`
    """
    if not (all(0 <= i <= s for i, s in zip(start_idx, img.shape)) and
            all(0 <= i <= s for i, s in zip(end_idx, img.shape))):
        return None
    return img[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]


def locate(raw_image, radius, signal_thresh, mass_thresh, bandpass=True,
           noise_radius=1):
    """Locate bright, Gaussian-like features in an image

    This implements an algorithm proposed by Crocker & Grier [Croc1996]_ and is
    based on the implementation by the Kilfoil group [Gao2009]_.

    This is the actual implementation. Usually, one would not call this
    directly but the wrapper functions :py:func:`api.locate` and
    :py:func:`api.batch`

    Parameters
    ----------
    raw_image : array-like
        Raw image data
    radius : int
        This should be a number a little greater than the radius of the
        peaks.
    signal_thresh : float
        A number roughly equal to the value of the brightest pixel (minus the
        CCD baseline) in the dimmest peak to be detected. Local maxima with
        brightest pixels below this threshold will be discarded.
    mass_thresh : float
        Use a number roughly equal to the integrated intensity (mass) of the
        dimmest peak (minus the CCD baseline) that should be detected. If this
        is too low more background will be detected. If it is too high more
        peaks will be missed.
    bandpass : bool, optional
        Set to True to turn on bandpass filtering, false otherwise. Default is
        True.
    noise_size : float, optional
        Noise correlation length in pixels. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Peak data. Column order is given by the ``col_nums`` attribute.
    """
    if bandpass:
        image = bandpass_filter(raw_image, radius, noise_radius, nonneg=True)
    else:
        image = raw_image

    peaks_found = find_peaks_yx(image, radius, signal_thresh)[:, ::-1]

    # create masks
    range_c = np.arange(-radius, radius + 1)
    range_sq = range_c**2
    # each entry is the distance from the center squared
    rsq_mask = range_sq[:, np.newaxis] + range_sq[np.newaxis, :]
    # boolean mask, circle with half the diameter radius (i. e. `radius` + 0.5)
    feat_mask = (rsq_mask <= (radius+0.5)**2)
    # each entry is the polar angle (however, in clockwise direction)
    theta_mask = np.arctan2(range_c[:, np.newaxis], range_c[np.newaxis, :])
    cos_mask = np.cos(2*theta_mask) * feat_mask
    cos_mask[radius, radius] = 0.
    sin_mask = np.sin(2*theta_mask) * feat_mask
    sin_mask[radius, radius] = 0.
    # x coordinate of every point
    x_mask = np.arange(2*radius + 1)[np.newaxis, :] * feat_mask
    # y coordinate of every point
    y_mask = x_mask.T

    # create output structure
    ret = np.empty((len(peaks_found), len(col_nums)))
    # boolean array that will tell us whether the estimated mass is greater
    # than mass_thresh
    valid = np.ones(len(peaks_found), dtype=bool)
    for i, (x, y) in enumerate(peaks_found):
        # region of interest for this peak
        roi = _get_crop(image, (y - radius, x - radius),
                        (y + radius + 1, x + radius + 1))
        if roi is None:
            valid[i] = False
            continue

        # estimate mass
        m = np.sum(roi * feat_mask)

        if m <= mass_thresh:
            # not bright enough, no further treatment
            valid[i] = False
            continue

        # estimate subpixel position by calculating center of mass
        # \sum_{i, j=0}^{2*radius} (i, j) * image(x+i, y+j)  then subtract the
        # coordinate of the center (i. e. radius)
        dx = np.sum(roi * x_mask) / m - radius
        dy = np.sum(roi * y_mask) / m - radius

        xc = x + dx
        yc = y + dy

        # Shift the image.
        # One extra pixel row/column is needed for interpolation.
        # E.g. if dx in [0, 1], the left ROI boundary does not need to be
        # shifted, the right needs to be shifted to the right by 1.
        # If dx in [-2, -1], the left ROI boundary needs to be shifted to the
        # left by 2, the right by 1.
        shift_x_int = math.floor(dx)
        shift_x_frac = dx % 1
        shift_y_int = math.floor(dy)
        shift_y_frac = dy % 1
        shifted_roi = _get_crop(
            image,
            (y-radius+shift_y_int, x-radius+shift_x_int),
            (y+radius+shift_y_int+2, x+radius+shift_x_int+2))
        if shifted_roi is None:
            valid[i] = False
            continue
        shifted_roi = scipy.ndimage.shift(
            shifted_roi.astype(float), (-shift_y_frac, -shift_x_frac), order=1,
            cval=np.nan)
        # Remove extra pixels
        shifted_roi = shifted_roi[:-1, :-1]

        # calculate peak properties
        # mass
        m = np.sum(shifted_roi * feat_mask)
        ret[i, col_nums.mass] = m

        # radius of gyration
        rg_sq = np.sum(shifted_roi * (rsq_mask * feat_mask + 1./6.))/m
        ret[i, col_nums.size] = np.sqrt(rg_sq)

        # eccentricity
        ecc = np.sqrt(np.sum(shifted_roi * cos_mask)**2 +
                      np.sum(shifted_roi * sin_mask)**2)
        ecc /= m - shifted_roi[radius, radius] + 1e-6
        ret[i, col_nums.ecc] = ecc

        # peak center, like above
        dx = np.sum(shifted_roi * x_mask) / m - radius
        dy = np.sum(shifted_roi * y_mask) / m - radius

        ret[i, col_nums.x] = xc + dx
        ret[i, col_nums.y] = yc + dy

    # remove peaks that are not brighter than mass_thresh or close to edges
    ret = ret[valid]
    return ret


def reconstruct(frames, radius=3, signal_thresh=300., mass_thresh=5000., bandpass=True, noise_radius=1):
    """Native batch-first frames -> table (1,N,6): frame,x,y,mass,size,ecc.
    N is inferred; row order is not a scientific correspondence requirement.
    """
    if frames.ndim!=4 or frames.shape[0]!=1:raise ValueError('Expected batch-first frames')
    rows=[]
    for i,frame in enumerate(frames[0]):
        result=locate(frame,radius,signal_thresh,mass_thresh,bandpass,noise_radius)
        rows.append(np.c_[np.full(len(result),i),result])
    return np.concatenate(rows,axis=0)[None].astype(np.float64)
