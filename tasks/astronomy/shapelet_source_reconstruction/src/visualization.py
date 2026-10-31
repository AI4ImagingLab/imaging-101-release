"""
Plotting utilities for shapelet source reconstruction results.

NOTE: Does NOT call matplotlib.use() — backend is set in main.py only.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_shapelet_decomposition(ngc_square, ngc_conv, ngc_resized, reconstructed,
                                 save_path=None):
    """Plot 4-panel shapelet decomposition figure.

    :param save_path: if provided, save figure to this path
    """
    f, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=False, sharey=False)
    for ax, img, title in zip(axes,
                              [ngc_square, ngc_conv, ngc_resized, reconstructed],
                              ["original image", "convolved image", "resized", "reconstructed"]):
        ax.matshow(img, origin='lower')
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lensing_stages(images, labels, save_path=None):
    """Plot 5-panel lensing simulation stages.

    :param images: list of 5 images (intrinsic, lensed, convolved, pixelized, noisy)
    :param labels: list of 5 label strings
    :param save_path: if provided, save figure to this path
    """
    f, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=False, sharey=False)
    for ax, img, label in zip(axes, images, labels):
        ax.matshow(img, origin='lower', extent=[0, 1, 0, 1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
        ax.text(0.05, 0.05, label, color="w", fontsize=20, transform=ax.transAxes)
    f.tight_layout()
    f.subplots_adjust(wspace=0., hspace=0.05)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction(panels, save_path=None):
    """Plot 2x3 reconstruction comparison figure.

    :param panels: list of 6 (image, title) tuples, arranged as 2 rows x 3 cols
    :param save_path: if provided, save figure to this path
    """
    f, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    for idx, (img, title) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        ax.matshow(img, origin='lower')
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_unlensed_stages(images, save_path=None):
    """Plot 4-panel deconvolution setup figure.

    :param images: list of 4 images (high-res, convolved, pixelized, noisy)
    :param save_path: if provided, save figure to this path
    """
    f, axes = plt.subplots(1, 4, figsize=(12, 4), sharex=False, sharey=False)
    for ax, img in zip(axes, images):
        ax.matshow(img, origin='lower', extent=[0, 1, 0, 1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.autoscale(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
