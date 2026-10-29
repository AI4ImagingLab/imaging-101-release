"""Visualization utilities and image quality metrics."""

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def psnr(ref, img):
    """Peak signal-to-noise ratio (PSNR).

    Parameters
    ----------
    ref : ndarray
        Reference (ground truth) image.
    img : ndarray
        Reconstructed image.

    Returns
    -------
    float
        PSNR in dB.
    """
    mse = np.mean((ref - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2):
    """Structural similarity index (single-channel).

    Parameters
    ----------
    img1, img2 : ndarray, shape (H, W)
        Single-channel images.

    Returns
    -------
    float
        SSIM value.
    """
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border=0):
    """Calculate mean SSIM across all spectral channels.

    Parameters
    ----------
    img1, img2 : ndarray, shape (H, W) or (H, W, nC)
        Input images.
    border : int
        Border pixels to exclude.

    Returns
    -------
    float
        Mean SSIM.
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def plot_spectral_bands(data, title, wavelength_start=400, wavelength_step=10,
                        vmin=None, vmax=None, save_path=None):
    """Plot all spectral bands in a grid.

    Parameters
    ----------
    data : ndarray, shape (H, W, nC)
        Spectral data cube.
    title : str
        Figure title.
    wavelength_start : int
        Starting wavelength in nm.
    wavelength_step : int
        Wavelength interval in nm.
    vmin, vmax : float, optional
        Color scale limits.
    save_path : str, optional
        Path to save the figure.
    """
    nC = data.shape[2]
    ncols = 8
    nrows = (nC + ncols - 1) // ncols

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 2.5))
    for i in range(nC):
        row, col = divmod(i, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.imshow(data[:, :, i], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax.set_title('{}nm'.format(wavelength_start + i * wavelength_step), fontsize=10)
        ax.axis('off')
    # hide unused cells
    for i in range(nC, nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(truth, recon, psnr_val, ssim_val, wavelength_start=400,
                    wavelength_step=10, save_path=None):
    """Plot ground truth vs reconstruction side by side for all bands.

    Parameters
    ----------
    truth, recon : ndarray, shape (H, W, nC)
        Ground truth and reconstruction.
    psnr_val, ssim_val : float
        Metrics for the title.
    save_path : str, optional
        Path to save the figure.
    """
    nC = truth.shape[2]
    vmin, vmax = truth.min(), truth.max()

    fig, axes = plt.subplots(2, nC, figsize=(48, 4))
    for i in range(nC):
        axes[0, i].imshow(truth[:, :, i], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        axes[0, i].set_title('{}nm'.format(wavelength_start + i * wavelength_step), fontsize=7)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[:, :, i], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('GT', fontsize=10, rotation=0, labelpad=25)
    axes[1, 0].set_ylabel('Recon', fontsize=10, rotation=0, labelpad=25)
    plt.suptitle('GT vs Reconstruction (PSNR {:.2f} dB, SSIM {:.4f})'.format(
        psnr_val, ssim_val), fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_measurement(meas, title='Measurement', save_path=None):
    """Plot the compressed measurement.

    Parameters
    ----------
    meas : ndarray, shape (H, W)
        2D measurement.
    save_path : str, optional
        Path to save the figure.
    """
    plt.figure()
    plt.imshow(meas, cmap='gray')
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
