"""Notebook comparisons using explicit local ODF evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import sample_odf


def comparison(signal, fresh, upstream):
    """Return signal, angular structure and upstream-error diagnostic panels."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    images = [signal[0, :, :, 0, :10].mean(-1), np.linalg.norm(fresh[0, :, :, 0, 1:], axis=-1), np.max(abs(fresh-upstream), axis=-1)[0, :, :, 0]]
    for ax, image, title in zip(axes, images, ['Mean unweighted signal (a.u.)', 'Anisotropic coefficient norm (sr^-1)', 'Maximum coefficient error (sr^-1)']):
        artist = ax.imshow(image.T, origin='lower'); ax.set_title(title); fig.colorbar(artist, ax=ax, shrink=.7)
    fig.tight_layout()
    return fig


def angular_comparison(fresh, upstream):
    """Compare fresh and original ODF curves in the image x-y plane."""
    angle = np.linspace(0, 2*np.pi, 361)
    directions = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], -1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), subplot_kw={'projection':'polar'})
    for ax, (x, y) in zip(axes, [(5,5),(12,12),(18,18)]):
        ax.plot(angle, sample_odf(upstream[0,x,y,0], directions), label='Original upstream')
        ax.plot(angle, sample_odf(fresh[0,x,y,0], directions), '--', label='Fresh local')
        ax.set_title(f'Voxel ({x},{y},0)'); ax.set_ylim(bottom=0)
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.1,1)); fig.tight_layout()
    return fig
