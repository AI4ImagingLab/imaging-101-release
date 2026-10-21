"""Review plots and independently implemented comparison metrics."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import DR, DZ


def comparison_metrics(estimate, reference):
    """Report cosine NCC and reference dynamic-range NRMSE; no alignment."""
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if (estimate.shape != reference.shape or estimate.size == 0 or np.iscomplexobj(estimate)
            or np.iscomplexobj(reference) or not np.isfinite(estimate).all()
            or not np.isfinite(reference).all() or np.ptp(reference) <= 0):
        raise ValueError('Expected finite real matching arrays and nonconstant reference')
    norm_product = np.linalg.norm(estimate)*np.linalg.norm(reference)
    return {'ncc': float(np.vdot(estimate, reference)/norm_product) if norm_product else 0.,
            'nrmse': float(np.sqrt(np.mean((estimate-reference)**2))/np.ptp(reference))}


def plot_comparison(projection, truth, upstream, fresh):
    """Return image comparison with physical axes and separately scaled residuals."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    extent = [0, 256*DR, 32*DZ, -32*DZ]
    maximum = float(np.max(truth))
    panels = [(projection[0], 'Observed projection', 'radiance', None, None),
              (truth[0], 'Synthetic truth', 'radiance/mm', 0, maximum),
              (upstream[0], 'Reproduced upstream: first order', 'radiance/mm', 0, maximum),
              (fresh[0], 'Fresh local: first order', 'radiance/mm', 0, maximum),
              ((fresh-truth)[0], 'Fresh minus truth', 'radiance/mm', -0.025, 0.025),
              ((fresh-upstream)[0], 'Fresh minus upstream', 'radiance/mm', -1e-12, 1e-12)]
    for ax, (data, title, unit, lower, upper) in zip(axes.flat, panels):
        im = ax.imshow(data, extent=extent, aspect='auto', cmap='coolwarm' if 'minus' in title else 'magma',
                       vmin=lower, vmax=upper)
        ax.set(title=title, xlabel='Impact y / radius r (mm)', ylabel='z (mm)')
        fig.colorbar(im, ax=ax, label=unit)
    return fig


def plot_profiles(truth, upstream, fresh, zero_order):
    """Compare fresh/upstream hold orders with independent truth at z=0."""
    r = np.arange(257)*DR
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for array, label, style in [(truth, 'Synthetic truth', '-'), (upstream, 'Upstream first order', '--'),
                                 (fresh, 'Fresh first order', ':'), (zero_order, 'Fresh zero order', '-.')]:
        ax.plot(r, array[0, 32], style, label=label, linewidth=1.5)
    ax.set(xlabel='r (mm)', ylabel='Emission (radiance/mm)', title='Radial emission profile at z = 0 mm')
    ax.legend()
    return fig


def plot_metric_bars(measurements):
    """Paired metric panels for explicitly named, actually computed results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    labels = list(measurements)
    for ax, metric in zip(axes, ['ncc', 'nrmse']):
        bars = ax.bar(labels, [measurements[label][metric] for label in labels], color=['#476c9b','#d57939','#3c8d7b','#9e6159'])
        ax.bar_label(bars, fmt='%.6f', fontsize=8, padding=3)
        ax.set(ylabel=metric.upper(), title=metric.upper()+' against synthetic truth')
        ax.tick_params(axis='x', labelrotation=20)
        ax.margins(y=0.2)
    return fig
