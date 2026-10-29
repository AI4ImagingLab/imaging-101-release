import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(mesh, ax=None, figsize=(6, 6)):
    """Plot the triangular mesh."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pts = mesh.node
    tri = mesh.element
    ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=0.5, color='gray')
    # Mark electrodes
    el_pts = pts[mesh.el_pos]
    ax.plot(el_pts[:, 0], el_pts[:, 1], 'ro', markersize=5)
    ax.set_aspect('equal')
    return ax

def plot_conductivity(mesh, perm, title="", ax=None, figsize=(6, 6)):
    """Plot conductivity distribution on mesh."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pts = mesh.node
    tri = mesh.element
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(perm), shading='flat')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax

def plot_greit_image(xg, yg, ds, title="", ax=None, figsize=(6, 6)):
    """Plot GREIT reconstruction on regular grid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.real(ds), interpolation='none', cmap=plt.cm.viridis)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax

def plot_reconstruction_comparison(reconstructions, ground_truths, meshes, metrics=None):
    """Plot side-by-side comparison of all reconstruction methods."""
    n_methods = len(reconstructions)
    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8), constrained_layout=True)

    for i, (name, recon) in enumerate(reconstructions.items()):
        mesh = meshes[name]
        gt = ground_truths[name]
        pts = mesh.node
        tri = mesh.element

        # Ground truth
        ax_gt = axes[0, i]
        ax_gt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(gt), shading='flat')
        ax_gt.set_aspect('equal')
        ax_gt.set_title(f'{name} - Ground Truth')

        # Reconstruction
        ax_rec = axes[1, i]
        if name == 'greit':
            xg, yg, ds = recon
            ax_rec.imshow(np.real(ds), interpolation='none', cmap=plt.cm.viridis)
        else:
            ax_rec.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(recon), shading='flat')
        ax_rec.set_aspect('equal')
        title = f'{name} - Reconstruction'
        if metrics and name in metrics:
            title += f'\nNRMSE={metrics[name].get("nrmse", 0):.3f}'
        ax_rec.set_title(title)

    return fig

def compute_metrics(reconstruction, ground_truth):
    """Compute reconstruction quality metrics.

    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed values (element-wise or grid)
    ground_truth : np.ndarray
        Ground truth values (same shape as reconstruction)

    Returns
    -------
    dict
        Dictionary with 'nrmse' and 'ncc' metrics
    """
    recon = np.real(reconstruction).ravel()
    gt = np.real(ground_truth).ravel()

    # Flux-normalize reconstruction to match ground truth
    if np.sum(np.abs(recon)) > 0:
        scale = np.sum(np.abs(gt)) / np.sum(np.abs(recon))
        recon_scaled = recon * scale
    else:
        recon_scaled = recon

    # NRMSE
    nrmse = np.linalg.norm(recon_scaled - gt) / np.linalg.norm(gt) if np.linalg.norm(gt) > 0 else float('inf')

    # NCC (normalized cross-correlation)
    gt_centered = gt - np.mean(gt)
    recon_centered = recon_scaled - np.mean(recon_scaled)
    ncc_num = np.dot(gt_centered, recon_centered)
    ncc_den = np.linalg.norm(gt_centered) * np.linalg.norm(recon_centered)
    ncc = ncc_num / ncc_den if ncc_den > 0 else 0.0

    return {"nrmse": float(nrmse), "ncc": float(ncc)}

def print_metrics_table(metrics):
    """Print a formatted metrics table."""
    print(f"{'Method':<20} {'NRMSE':>10} {'NCC':>10}")
    print("-" * 42)
    for name, m in metrics.items():
        print(f"{name:<20} {m['nrmse']:>10.4f} {m['ncc']:>10.4f}")
