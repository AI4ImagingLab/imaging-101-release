"""Maintainer comparisons of density, phase, relative strain and observations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from .physics_model import raw_observation_intensity, projected_strain
from .preprocessing import real_space_basis, reciprocal_geometry, prepare_observations


def support_boundary_figure(reconstruction, maximum_points=15000):
    """Compare the paper's three density levels using sampled boundary voxels.

    This is explicitly a voxel-boundary point rendering, not a triangulated
    subvoxel isosurface. Reconstruction arrays are never downsampled or changed.
    """
    from scipy.ndimage import binary_erosion
    figure = plt.figure(figsize=(14, 9))
    basis = real_space_basis()
    for position, obj in enumerate(reconstruction[0]):
        amplitude = abs(obj)
        for column, level in enumerate((np.exp(-1), .5, .6)):
            mask = amplitude >= level * amplitude.max()
            boundary = mask & ~binary_erosion(mask)
            indices = np.asarray(np.nonzero(boundary))
            step = max(1, int(np.ceil(indices.shape[1]/maximum_points)))
            indices = indices[:, ::step]
            coordinates = basis @ (indices-np.asarray(obj.shape)[:, None]/2) / 1000.
            axis = figure.add_subplot(2, 3, position*3+column+1, projection="3d")
            axis.scatter(*coordinates, s=.5, alpha=.4, color="teal", rasterized=True)
            axis.set(xlabel="outboard (um)", ylabel="up (um)", zlabel="downstream (um)",
                     title=f"P{position+1}: amplitude >= {level:.3g} maximum")
            axis.set_box_aspect(np.maximum(np.ptp(coordinates, axis=1), np.finfo(float).eps))
    figure.suptitle("Three physical density-level comparisons; sampled boundary voxel centers")
    figure.tight_layout()
    return figure


def density_phase_figures(reconstruction, label="Fresh cleaned reconstruction"):
    """Return separate 2-by-3 amplitude and masked phase figures for P1/P2.

    Input has the submission shape (1,2,126,256,256); slice labels retain
    oblique array axes rather than misrepresenting an orthogonal laboratory grid.
    """
    objects = reconstruction[0]
    amplitude_figure, amplitude_axes = plt.subplots(2, 3, figsize=(13, 8))
    phase_figure, phase_axes = plt.subplots(2, 3, figsize=(13, 8))
    for position, obj in enumerate(objects):
        absolute = abs(obj)
        peak = np.unravel_index(absolute.argmax(), obj.shape)
        for axis in range(3):
            view = amplitude_axes[position, axis].imshow(
                absolute.max(axis=axis), origin="lower", cmap="magma")
            amplitude_axes[position, axis].set_title(f"P{position+1}: maximum over axis {axis}")
            amplitude_figure.colorbar(view, ax=amplitude_axes[position, axis],
                                     label="density amplitude (sqrt count)")
            selector = [slice(None)] * 3
            selector[axis] = peak[axis]
            plane = obj[tuple(selector)]
            masked = np.ma.masked_where(abs(plane) < .1 * absolute.max(), np.angle(plane))
            view = phase_axes[position, axis].imshow(
                masked, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
            phase_axes[position, axis].set_title(f"P{position+1}: axis {axis} slice at density-peak index {peak[axis]}")
            phase_figure.colorbar(view, ax=phase_axes[position, axis], label="phase (rad)")
    for figure, suffix in ((amplitude_figure, "amplitude"), (phase_figure, "phase")):
        figure.suptitle(f"{label}: {suffix}; oblique voxel indices")
        figure.tight_layout()
    return {"amplitude": amplitude_figure, "phase": phase_figure}


def strain_figure(reconstruction):
    """Plot relative projected strain on oblique slices through density maximum."""
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    basis = real_space_basis()
    _, q = reciprocal_geometry()
    for position, obj in enumerate(reconstruction[0]):
        peak = np.unravel_index(abs(obj).argmax(), obj.shape)
        strain, valid = projected_strain(obj, basis, q)
        limit = np.percentile(abs(strain[valid]), 99) if valid.any() else 1.
        limit = max(float(limit), np.finfo(float).eps)
        for axis in range(3):
            selector = [slice(None)] * 3
            selector[axis] = peak[axis]
            selector = tuple(selector)
            view = axes[position, axis].imshow(
                np.ma.masked_where(~valid[selector], strain[selector]),
                origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[position, axis].set_title(f"P{position+1}: axis {axis} slice at density-peak index {peak[axis]}")
            figure.colorbar(view, ax=axes[position, axis], label="relative projected strain (1)")
    figure.suptitle("Nominal geometry; interior mean removed; two-step phase difference < pi required")
    figure.tight_layout()
    return figure


def observation_figure(reconstruction, coherence, counts):
    """Compare three raw121 count projections and signed residuals for each position."""
    figure, axes = plt.subplots(6, 3, figsize=(14, 22))
    for position in range(2):
        _, metadata = prepare_observations(counts[position])
        observed = counts[position].astype(np.float64)
        predicted = raw_observation_intensity(reconstruction[0, position], coherence[0, position],
                                            metadata['peak_shift'], metadata['padding'])
        for axis in range(3):
            observed_plane, predicted_plane = observed.sum(axis=axis), predicted.sum(axis=axis)
            extent = max(float(observed_plane.max()), float(predicted_plane.max()), 1.01)
            for row, (plane, title) in enumerate(((observed_plane, "measured"),
                                                  (predicted_plane, "predicted"))):
                panel = axes[3*position+row, axis]
                view = panel.imshow(np.maximum(plane, 1e-12), origin="lower",
                                    norm=LogNorm(vmin=1., vmax=extent), cmap="magma")
                panel.set_title(f"P{position+1}: {title}, sum over axis {axis}")
                figure.colorbar(view, ax=panel, label="count")
            residual = predicted_plane - observed_plane
            limit = max(float(abs(residual).max()), 1.01)
            panel = axes[3*position+2, axis]
            view = panel.imshow(residual, origin="lower", cmap="RdBu_r",
                                norm=SymLogNorm(linthresh=1., vmin=-limit, vmax=limit))
            panel.set_title(f"P{position+1}: predicted minus measured, sum axis {axis}")
            figure.colorbar(view, ax=panel, label="count (signed symlog)")
    figure.suptitle("All original121 measured planes; inverse roll/crop after finite blur; no threshold mask")
    figure.tight_layout()
    return figure


def coherence_figure(coherence):
    """Plot all three central planes of each inferred unit-mass coherence kernel."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for position, kernel in enumerate(coherence[0]):
        for axis in range(3):
            view = axes[position, axis].imshow(np.take(kernel, kernel.shape[axis]//2, axis=axis),
                                               origin="lower", cmap="cividis")
            axes[position, axis].set_title(f"P{position+1}: kernel plane normal to axis {axis}")
            figure.colorbar(view, ax=axes[position, axis], label="probability mass per voxel")
    figure.tight_layout()
    return figure


def ensemble_error_figure(upstream_errors, fresh_errors):
    """Paired start-by-start objective comparisons; neither series is historical."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for position, axis in enumerate(axes):
        axis.plot(upstream_errors[position], label="reproduced corrected upstream", linewidth=1.)
        axis.plot(fresh_errors[position], label="fresh cleaned method", linewidth=1., alpha=.7)
        axis.set(xlabel="random-start index", ylabel="final squared relative amplitude error",
                 title=f"P{position+1}: all 200 starts")
        axis.legend()
    figure.tight_layout()
    return figure
