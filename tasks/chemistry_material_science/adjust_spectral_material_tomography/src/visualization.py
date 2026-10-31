"""Scientific comparison plots and explicitly defined numerical metrics."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def align_components(spectra, reference_spectra):
    """Resolve only component permutation through normalized spectral distance."""
    x = np.asarray(spectra, dtype=np.float64)
    ref = np.asarray(reference_spectra, dtype=np.float64)
    xn, rn = np.linalg.norm(x, axis=1), np.linalg.norm(ref, axis=1)
    if np.any(xn == 0) or np.any(rn == 0):
        raise ValueError('component alignment requires nonzero spectra')
    similarity = (x @ ref.T) / (xn[:, None] * rn[None, :])
    estimate, reference = linear_sum_assignment(1 - similarity)
    order = np.empty(len(ref), dtype=int)
    order[reference] = estimate
    return order


def numeric_metrics(estimate, reference):
    """Cosine NCC and RMS error divided by reference dynamic range."""
    x, r = np.asarray(estimate).ravel(), np.asarray(reference).ravel()
    denominator = np.linalg.norm(x) * np.linalg.norm(r)
    dynamic_range = np.ptp(r)
    if denominator == 0 or dynamic_range == 0:
        raise ValueError('NCC/NRMSE undefined for a zero norm or constant reference')
    return {'ncc': float(np.dot(x, r) / denominator),
            'nrmse': float(np.sqrt(np.mean((x - r) ** 2)) / dynamic_range)}


def plot_acquisition(observations, corrected_data):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    image = axes[0].imshow(observations['counts'][0], origin='lower', aspect='auto', cmap='magma')
    axes[0].set(title='Measured photon counts: acquired angle0', xlabel='Detector position', ylabel='Energy channel')
    figure.colorbar(image, ax=axes[0], label='Photon counts')
    sino = corrected_data[:, 61].reshape(257, 600).T
    image = axes[1].imshow(sino, aspect='auto', cmap='viridis')
    axes[1].set(title='Corrected spectral sinogram: channel62', xlabel='Detector position', ylabel='Reversed-angle index')
    figure.colorbar(image, ax=axes[1], label='Optical depth')
    axes[2].plot(observations['energies'], observations['flat'].mean(axis=1), label='Flat')
    axes[2].plot(observations['energies'], observations['counts'].mean(axis=(0, 2)), label='Specimen mean')
    axes[2].set(title='Actual spectral acquisition', xlabel='Energy-bin midpoint (keV)', ylabel='Photon counts')
    axes[2].legend()
    return figure


def plot_reference_comparison(submission, reference):
    order = align_components(submission['spectra'][0], reference['spectra'][0])
    images = submission['reconstruction'][0, order]
    reference_images = reference['reconstruction'][0]
    figure, axes = plt.subplots(4, 3, figsize=(11, 13), constrained_layout=True)
    for k in range(4):
        scale = max(float(reference_images[k].max()), float(images[k].max()))
        for j, (array, title) in enumerate(((reference_images[k], 'Independent corrected oracle'),
                                           (images[k], 'Fresh cleaned ADJUST'))):
            plot = axes[k, j].imshow(array, vmin=0, vmax=scale, cmap='inferno')
            axes[k, j].set_title(f'Component{k+1}: {title}', fontsize=9)
            figure.colorbar(plot, ax=axes[k, j], label='Abundance weight')
        error = images[k] - reference_images[k]
        limit = max(float(np.max(abs(error))), 1e-12)
        plot = axes[k, 2].imshow(error, cmap='coolwarm', vmin=-limit, vmax=limit)
        axes[k, 2].set_title(f"Difference; NRMSE={numeric_metrics(images[k],reference_images[k])['nrmse']:.3g}", fontsize=9)
        figure.colorbar(plot, ax=axes[k, 2], label='Abundance error')
        for axis in axes[k]:
            axis.set(xlabel='Column', ylabel='Row')
    return figure


def plot_methods(methods, reference_spectra):
    figure, axes = plt.subplots(4, len(methods), figsize=(4 * len(methods), 12), constrained_layout=True)
    for column, (name, result) in enumerate(methods.items()):
        order = align_components(result['spectra'], reference_spectra)
        for row, component in enumerate(order):
            image = result['maps'][:, component].reshape(255, 255, order='F')
            plot = axes[row, column].imshow(image, cmap='inferno')
            axes[row, column].set_title(f'{name}: component{row+1}', fontsize=9)
            figure.colorbar(plot, ax=axes[row, column], label='Method-specific abundance scale')
    return figure


def plot_spectral_results(energies, submission, reference, history, reference_history):
    order = align_components(submission['spectra'][0], reference['spectra'][0])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for k, index in enumerate(order):
        line, = axes[0, 0].plot(energies[29:94], reference['spectra'][0, k], label=f'Upstream{k+1}')
        axes[0, 0].plot(energies[29:94], submission['spectra'][0, index], '--', color=line.get_color(), label=f'Fresh{k+1}')
    axes[0, 0].set(xlabel='Energy-bin midpoint (keV)', ylabel='Optical depth per image pixel', title='Spectral scale retained')
    axes[0, 0].legend(ncol=2, fontsize=7)
    plot = axes[0, 1].imshow(submission['dictionary_weights'][0, order], aspect='auto', cmap='viridis')
    axes[0, 1].set(xlabel='Dictionary atom (last = active quartz estimate)', ylabel='Aligned component', title='Fresh dictionary coefficients')
    figure.colorbar(plot, ax=axes[0, 1], label='Dimensionless weight')
    axes[1, 0].plot(reference_history[:, 0], label='Independent corrected oracle')
    axes[1, 0].plot(history[:, 0], '--', label='Fresh cleaned')
    axes[1, 0].set(xlabel='Outer iteration', ylabel='Relative data residual', title='Nonmonotone dynamics; best iterate is returned')
    axes[1, 0].legend()
    axes[1, 1].semilogy(history[:, 3])
    axes[1, 1].set(xlabel='Outer iteration', ylabel='Residual-lift coefficient rho', title='Actual dynamic acceleration')
    return figure
