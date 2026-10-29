"""
Preprocessing module for Lucky Imaging
=======================================

Load video frames, convert to monochrome, apply Gaussian blur,
compute brightness statistics, and prepare Laplacian images.

Functions
---------
load_frames           Load frames from raw_data.npz
to_mono               Convert RGB frame to monochrome
gaussian_blur         Apply Gaussian blur and upscale to 16-bit
average_brightness    Compute mean brightness within valid range
compute_laplacian     Compute Laplacian of blurred frame at stride
prepare_all_frames    Prepare all frames for the pipeline
"""

import os
import json

import cv2
import numpy as np


def load_frames(data_dir="data"):
    """Load video frames from raw_data.npz.

    Parameters
    ----------
    data_dir : str -- path to data directory

    Returns
    -------
    frames : ndarray, shape (N, H, W, 3), dtype uint8 -- RGB video frames
    meta : dict -- metadata from meta_data.json
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))
    frames = raw["frames"]
    # Remove batch dimension if present
    if frames.ndim == 5:
        frames = frames[0]

    with open(os.path.join(data_dir, "meta_data.json"), "r") as f:
        meta = json.load(f)

    return frames, meta


def to_mono(frame):
    """Convert an RGB frame to monochrome (luminance).

    Parameters
    ----------
    frame : ndarray, shape (H, W, 3), dtype uint8

    Returns
    -------
    mono : ndarray, shape (H, W), dtype uint8
    """
    if frame.ndim == 2:
        return frame
    # Convert RGB to grayscale using OpenCV (expects BGR, but the luminance
    # weights are close enough for RGB too; for exact match convert to BGR first)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def gaussian_blur(frame_mono, gauss_width=7):
    """Apply Gaussian blur and upscale to 16-bit for precision.

    Following PSS: upscale uint8 to uint16 (*256) before blurring to preserve
    dynamic range during the Gaussian convolution.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W), dtype uint8
    gauss_width : int -- kernel size (must be odd)

    Returns
    -------
    blurred : ndarray, shape (H, W), dtype uint16
    """
    upscaled = frame_mono.astype(np.uint16) * 256
    blurred = cv2.GaussianBlur(upscaled, (gauss_width, gauss_width), 0)
    return blurred.astype(np.uint16)


def average_brightness(frame_mono, low=16, high=240):
    """Compute mean brightness within the [low, high] range.

    Parameters
    ----------
    frame_mono : ndarray, shape (H, W), dtype uint8
    low : int -- lower brightness threshold
    high : int -- upper brightness threshold

    Returns
    -------
    brightness : float -- mean brightness of thresholded pixels
    """
    # Threshold to zero pixels outside [low, high]
    _, threshed = cv2.threshold(frame_mono, low, 0, cv2.THRESH_TOZERO)
    _, threshed = cv2.threshold(threshed, high, 0, cv2.THRESH_TOZERO_INV)
    mean_val = cv2.mean(threshed)[0]
    return max(mean_val, 1e-10)


def compute_laplacian(frame_blurred, stride=2):
    """Compute the Laplacian of the blurred frame at given stride.

    Parameters
    ----------
    frame_blurred : ndarray, shape (H, W), dtype uint16
    stride : int -- downsampling stride

    Returns
    -------
    lap : ndarray, dtype uint8 -- absolute Laplacian at reduced resolution
    """
    subsampled = frame_blurred[::stride, ::stride]
    lap = cv2.Laplacian(subsampled, cv2.CV_32F)
    # Convert to uint8 with scale factor 1/256 (to undo the 16-bit upscale)
    lap_abs = cv2.convertScaleAbs(lap, alpha=1.0 / 256.0)
    return lap_abs


def prepare_all_frames(frames, gauss_width=7, stride=2):
    """Prepare all frames: mono, blurred, brightness, Laplacian.

    Parameters
    ----------
    frames : ndarray, shape (N, H, W, 3), dtype uint8
    gauss_width : int -- Gaussian blur kernel size
    stride : int -- downsampling stride for Laplacian

    Returns
    -------
    frames_data : dict with keys:
        'frames_rgb' : ndarray (N, H, W, 3) uint8 -- original frames
        'mono' : list of ndarray (H, W) uint8
        'blurred' : list of ndarray (H, W) uint16
        'brightness' : ndarray (N,) float64
        'laplacian' : list of ndarray (H//stride, W//stride) uint8
        'n_frames' : int
        'shape' : tuple (H, W)
    """
    n_frames = frames.shape[0]

    mono_list = []
    blurred_list = []
    brightness_arr = np.zeros(n_frames, dtype=np.float64)
    laplacian_list = []

    for i in range(n_frames):
        mono = to_mono(frames[i])
        blurred = gaussian_blur(mono, gauss_width)
        bright = average_brightness(mono)
        lap = compute_laplacian(blurred, stride)

        mono_list.append(mono)
        blurred_list.append(blurred)
        brightness_arr[i] = bright
        laplacian_list.append(lap)

    return {
        'frames_rgb': frames,
        'mono': mono_list,
        'blurred': blurred_list,
        'brightness': brightness_arr,
        'laplacian': laplacian_list,
        'n_frames': n_frames,
        'shape': (frames.shape[1], frames.shape[2]),
    }
