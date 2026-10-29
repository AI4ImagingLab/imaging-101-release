"""Maintainer metrics and scientific comparison figures; no backend mutation."""
import numpy as np
import matplotlib.pyplot as plt


def array_metrics(estimate, reference):
    """Cosine NCC and dynamic-range NRMSE; explicitly handle zero estimates."""
    x, y = np.asarray(estimate, np.float64).ravel(), np.asarray(reference, np.float64).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Metric inputs must have identical shapes and finite values')
    span = float(np.ptp(y))
    norm = float(np.linalg.norm(y))
    if span <= 0 or norm <= 0:
        raise ValueError('Reference is degenerate for NCC/NRMSE')
    cosine = float(np.dot(x, y)/(np.linalg.norm(x)*norm)) if np.linalg.norm(x)>0 else 0.
    return {'ncc': cosine, 'nrmse': float(np.sqrt(np.mean((x-y)**2))/span)}


def compare_outputs(output, reference, mask):
    """Evaluate each required quantity using the fixed observation-validity mask."""
    return {key: array_metrics(output[key][..., mask], reference[key][..., mask]) for key in reference}


def plot_stages(data, output, diagnostics, upstream):
    """Six complete-frame stage comparisons at the final acquisition."""
    stages = [output['network_timeseries'][-1], diagnostics['troposphere_corrected'][-1],
              diagnostics['deramped'][-1], output['reconstruction'][-1],
              upstream['reconstruction'][-1], output['reconstruction'][-1]-upstream['reconstruction'][-1]]
    titles = ['Fresh weighted SBAS LOS', 'Fresh minus ERA5', 'Fresh minus spatial ramp',
              'Fresh minus DEM residual', 'Reproduced upstream final LOS', 'Fresh - upstream']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for i, (ax, image, title) in enumerate(zip(axes.flat, stages, titles)):
        displayed = np.ma.masked_where(~data['analysis_mask'], image)
        limit = max(float(np.max(np.abs(displayed))), 1e-8) if i==5 else .3
        artist = ax.imshow(displayed, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_title(title); ax.set_xlabel('Range pixel'); ax.set_ylabel('Azimuth pixel')
        fig.colorbar(artist, ax=ax, label='LOS displacement (m)')
    return fig


def plot_parameters(data, output, diagnostics):
    """Quality, integer region, DEM and eruption component maps."""
    items = [(output['temporal_coherence'], 'Temporal coherence', '1', 'viridis'),
             (output['dem_error'], 'Estimated DEM residual', 'm', 'RdBu_r'),
             (output['deformation_parameters'][1], 'Deformation velocity', 'm/year', 'RdBu_r'),
             (output['deformation_parameters'][2], 'Deformation acceleration', 'm/year squared', 'RdBu_r'),
             (output['deformation_parameters'][3], '2017 eruption step', 'm', 'RdBu_r'),
             (output['deformation_parameters'][4], '2018 eruption step', 'm', 'RdBu_r')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (image, title, unit, cmap) in zip(axes.flat, items):
        artist = ax.imshow(np.ma.masked_where(~data['analysis_mask'], image), cmap=cmap)
        ax.set_title(title); ax.set_xlabel('Range pixel'); ax.set_ylabel('Azimuth pixel')
        fig.colorbar(artist, ax=ax, label=unit)
    return fig


def plot_traces(output, upstream, diagnostics, dates, coordinates):
    """Upstream/fresh traces, complete-field residual and metric comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for y, x in coordinates:
        label = f'pixel ({y}, {x})'
        axes[0, 0].plot(dates, output['reconstruction'][:, y, x], label=label)
        axes[0, 0].plot(dates, upstream['reconstruction'][:, y, x], '--', linewidth=.8)
        axes[0, 1].plot(dates, diagnostics['residual'][:, y, x], label=label)
    axes[0, 0].set_title('Fresh (solid) / reproduced upstream (dashed)'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title('Fresh temporal model residual')
    axes[0, 0].set_ylabel('LOS displacement (m)'); axes[0, 1].set_ylabel('Residual (m)')
    for ax in axes[0]:
        ax.tick_params(axis='x', rotation=25); ax.set_xlabel('Acquisition date')
    image = axes[1, 0].imshow(np.sqrt(np.mean(diagnostics['residual']**2, axis=0))*1000, cmap='magma', vmin=0, vmax=30)
    fig.colorbar(image, ax=axes[1, 0], label='Temporal RMS (mm)'); axes[1, 0].set_title('Full-scene model misfit (not ground-truth error)')
    diff = output['reconstruction'] - upstream['reconstruction']
    axes[1, 1].plot(np.max(np.abs(diff), axis=(1, 2))*1000)
    axes[1, 1].set_title('Fresh / reproduced upstream numerical agreement')
    axes[1, 1].set_xlabel('Acquisition index'); axes[1, 1].set_ylabel('Maximum absolute difference (mm)')
    return fig
