"""Maintainer comparison figure; backend is selected by the caller."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import green_strain


def plot_comparison(reference, deformed, nodes, elements, displacement, upstream):
    """Show observations, displacement comparisons and derived strain."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
    for ax, image, title in zip(axes[0, :2], [reference, deformed], ['Reference speckle', 'Deformed speckle']):
        ax.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.set_xlabel('x / pixel'); ax.set_ylabel('y / pixel')
    axes[0, 2].quiver(nodes[:, 0], nodes[:, 1], displacement[:, 0], displacement[:, 1],
                      angles='xy', scale_units='xy', scale=.04)
    axes[0, 2].set_title('Fresh nodal displacement (25x arrows)')
    axes[0, 2].set_aspect('equal')
    axes[0, 2].set(xlim=(20,240), ylim=(0,460), xlabel='x / pixel', ylabel='y / pixel')
    for component, label in enumerate(['x', 'y']):
        ax = axes[1, component]
        ax.plot(upstream[:, component], 'o-', label='Reproduced upstream')
        ax.plot(displacement[:, component], 'x--', label='Fresh local')
        ax.set(xlabel='Node index', ylabel=f'u{label} / pixel', title=f'{label} displacement comparison')
        ax.legend()
    strain = green_strain(nodes, nodes+displacement, elements)
    grid = nodes.reshape(5,3,2)
    scatter = axes[1, 2].pcolormesh(grid[:,:,0], grid[:,:,1], strain[:,1,1].reshape(4,2), shading='flat')
    fig.colorbar(scatter, ax=axes[1, 2], label='Eyy / dimensionless')
    axes[1, 2].set(title='Derived center Green-Lagrange strain', xlabel='x / pixel', ylabel='y / pixel')
    return fig
