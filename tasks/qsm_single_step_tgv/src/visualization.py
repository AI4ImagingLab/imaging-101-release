"""Maintainer plots and named, explicit numerical comparisons."""
import numpy as np
import matplotlib.pyplot as plt


def metrics(estimate, reference, mask):
    """Cosine NCC, reference-range NRMSE, and explicitly named L2 error."""
    a, b = estimate[mask].ravel(), reference[mask].ravel()
    if not len(a) or np.linalg.norm(b) == 0 or np.ptp(b) == 0:
        raise ValueError('Nondegenerate finite arrays required')
    return dict(ncc=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a) else 0.,
                nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)),
                relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b)))


def observation_metrics(reconstruction, residual_phase, phase, mask, spacing):
    """PDE constraint and residual-potential energy in the acquired physical units."""
    from .physics_model import interior_mask, laplacian, wave_operator, wrapped_laplacian
    from .preprocessing import PHASE_PER_PPM
    mi = interior_mask(mask)
    lphi = wrapped_laplacian(phase, spacing)
    residual = laplacian(residual_phase, spacing)-wave_operator(reconstruction*PHASE_PER_PPM, spacing)-lphi
    return {'pde_relative_residual': float(np.linalg.norm(residual[mi])/np.linalg.norm(lphi[mi])),
            'eta_rms_rad': float(np.sqrt(np.mean(residual_phase[mask]**2)))}


def observation_figure(observations):
    """Central axial measurements; display copies never alter reconstruction."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, unit in zip(axes, ('phase', 'magnitude', 'mask'), ('rad', 'a.u.', 'binary')):
        im = ax.imshow(observations[key][:, :, 80].T, origin='lower', cmap='gray')
        ax.set_title(f'{key}: axial z=80 [{unit}]')
        ax.set_xlabel('x voxel'); ax.set_ylabel('y voxel')
        fig.colorbar(im, ax=ax, shrink=.8)
    return fig


def reconstruction_figure(reconstruction, reference, mask, reference_name='COSMOS'):
    """Matched orthogonal slices and a signed difference, all in ppm."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, axis in enumerate(range(3)):
        arrays = (reference, reconstruction, reconstruction-reference)
        for col, a in enumerate(arrays):
            index = a.shape[axis]//2
            plane = np.take(np.where(mask, a, np.nan), index, axis=axis).T
            im = axes[row, col].imshow(plane, origin='lower', cmap='coolwarm', vmin=-.15, vmax=.15)
            axes[row, col].set_title(f'{(reference_name, "Fresh SSTGV", "Fresh - reference")[col]}, axis {axis}={index}')
            axes[row, col].set_xlabel('voxel'); axes[row, col].set_ylabel('voxel')
            fig.colorbar(im, ax=axes[row, col], label='ppm', shrink=.75)
    return fig


def method_comparison_figure(arrays, names, reference, mask):
    """Scientific method comparison: matched images and paired metric bars."""
    fig, axes = plt.subplots(2, len(arrays), figsize=(4*len(arrays), 7), constrained_layout=True)
    scores = []
    for col, (a, name) in enumerate(zip(arrays, names)):
        im = axes[0, col].imshow(np.where(mask[:, :, 80], a[:, :, 80], np.nan).T,
                                origin='lower', cmap='coolwarm', vmin=-.15, vmax=.15)
        axes[0, col].set_title(name); axes[0, col].set_xlabel('x voxel')
        fig.colorbar(im, ax=axes[0, col], label='ppm', shrink=.75)
        scores.append(metrics(a, reference, mask))
        axes[1, col].bar(['NCC', 'L2 error'], [scores[-1]['ncc'], scores[-1]['relative_l2']], color=['#277da8', '#d77a38'])
        axes[1, col].set_ylim(0, max(1.5, scores[-1]['relative_l2']*1.15))
        axes[1, col].set_title('Independent COSMOS comparison')
        for i, value in enumerate([scores[-1]['ncc'], scores[-1]['relative_l2']]):
            axes[1, col].text(i, value+.02, f'{value:.4f}', ha='center')
    return fig, scores
