"""Maintainer plots. Importing this module does not change the plotting backend."""
import numpy as np
import matplotlib.pyplot as plt


def observation_figure(hologram: np.ndarray, vacuum: np.ndarray):
    """Return object/vacuum intensity and log-Fourier-power figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, (name, image) in enumerate((('Fe needle', hologram[0]), ('Vacuum calibration', vacuum[0]))):
        im = axes[row, 0].imshow(image, cmap='gray', origin='upper')
        axes[row, 0].set_title(name + ': measured intensity')
        fig.colorbar(im, ax=axes[row, 0], label='detector counts')
        spec = np.log10(1 + np.abs(np.fft.fftshift(np.fft.fft2(image))))
        im = axes[row, 1].imshow(spec, cmap='magma', origin='upper')
        axes[row, 1].set_title(name + ': log10(1 + |DFT|)')
        fig.colorbar(im, ax=axes[row, 1], label='log Fourier amplitude')
    for ax in axes.ravel(): ax.set(xlabel='x index (pixel)', ylabel='y index (pixel)')
    return fig


def comparison_figure(fresh: np.ndarray, upstream: np.ndarray, uncorrected: np.ndarray,
                      weak_aperture: np.ndarray):
    """Return amplitude, principal phase, independent parity and bandwidth comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        (np.abs(fresh[0]), 'Fresh corrected amplitude', 'viridis', 'dimensionless'),
        (np.angle(fresh[0]), 'Fresh corrected principal phase', 'twilight', 'rad'),
        (np.abs(fresh[0] - upstream[0]), 'Fresh minus original-code wave', 'inferno', 'absolute complex error'),
        (np.angle(uncorrected), 'Fresh wave without vacuum correction', 'twilight', 'rad'),
        (np.angle(weak_aperture), 'Fresh weak-object aperture (2/3 radius)', 'twilight', 'rad'),
        (np.angle(fresh[0] * np.conj(weak_aperture)), 'Selected minus smaller-aperture phase', 'coolwarm', 'rad')]
    for ax, (image, title, cmap, label) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, origin='upper')
        ax.set(title=title, xlabel='x index (pixel)', ylabel='y index (pixel)')
        fig.colorbar(im, ax=ax, label=label)
    return fig
