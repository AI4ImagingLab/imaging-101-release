"""
Visualization and metrics for BH-NeRF task.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def compute_metrics(estimate, ground_truth):
    """
    Compute 3D emission recovery metrics.

    Parameters
    ----------
    estimate : np.ndarray
        Estimated emission (arbitrary shape).
    ground_truth : np.ndarray
        Ground truth emission (same shape).

    Returns
    -------
    metrics : dict with keys 'nrmse', 'ncc', 'psnr'
    """
    est = estimate.ravel().astype(np.float64)
    gt = ground_truth.ravel().astype(np.float64)

    mse = np.mean((est - gt) ** 2)
    nrmse = np.sqrt(mse) / (np.max(gt) - np.min(gt) + 1e-30)

    est_centered = est - est.mean()
    gt_centered = gt - gt.mean()
    ncc = float(np.dot(est_centered, gt_centered) /
                (np.linalg.norm(est_centered) * np.linalg.norm(gt_centered) + 1e-30))

    psnr = float(10.0 * np.log10(np.max(gt) ** 2 / (mse + 1e-30)))

    return {
        'nrmse': float(nrmse),
        'ncc': float(ncc),
        'psnr': float(psnr),
    }


def compute_image_metrics(pred_movie, true_movie):
    """
    Compute image-plane metrics averaged over time frames.

    Parameters
    ----------
    pred_movie : np.ndarray, shape (n_frames, H, W)
    true_movie : np.ndarray, shape (n_frames, H, W)

    Returns
    -------
    metrics : dict with keys 'nrmse_image', 'ncc_image', 'lightcurve_mse'
    """
    nrmse_list = []
    ncc_list = []
    for i in range(len(pred_movie)):
        m = compute_metrics(pred_movie[i], true_movie[i])
        nrmse_list.append(m['nrmse'])
        ncc_list.append(m['ncc'])

    pred_lc = pred_movie.sum(axis=(-1, -2))
    true_lc = true_movie.sum(axis=(-1, -2))
    lc_mse = float(np.mean((pred_lc - true_lc) ** 2))

    return {
        'nrmse_image': float(np.mean(nrmse_list)),
        'ncc_image': float(np.mean(ncc_list)),
        'lightcurve_mse': lc_mse,
    }


def plot_emission_slices(emission_3d, fov_M, ground_truth=None,
                         save_path=None):
    """
    Plot x-y, x-z, y-z slices through the 3D emission volume.

    Parameters
    ----------
    emission_3d : np.ndarray, shape (D, H, W)
    fov_M : float
    ground_truth : np.ndarray, optional
    save_path : str, optional
    """
    mid = emission_3d.shape[0] // 2
    extent = [-fov_M / 2, fov_M / 2, -fov_M / 2, fov_M / 2]

    if ground_truth is not None:
        gt_mid = min(mid, ground_truth.shape[0] // 2)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        titles = ['x-y (GT)', 'x-z (GT)', 'y-z (GT)',
                  'x-y (Est)', 'x-z (Est)', 'y-z (Est)']
        slices_gt = [ground_truth[:, :, gt_mid], ground_truth[:, gt_mid, :],
                     ground_truth[gt_mid, :, :]]
        slices_est = [emission_3d[:, :, mid], emission_3d[:, mid, :],
                      emission_3d[mid, :, :]]
        vmax = max(np.max(ground_truth), np.max(emission_3d))
        for i in range(3):
            im = axes[0, i].imshow(slices_gt[i].T, origin='lower',
                                   extent=extent, cmap='afmhot', vmin=0,
                                   vmax=vmax)
            axes[0, i].set_title(titles[i])
            axes[0, i].set_xlabel('x [M]' if i == 0 else ('x [M]' if i == 1 else 'y [M]'))
            axes[0, i].set_ylabel('y [M]' if i == 0 else 'z [M]')
            im = axes[1, i].imshow(slices_est[i].T, origin='lower',
                                   extent=extent, cmap='afmhot', vmin=0,
                                   vmax=vmax)
            axes[1, i].set_title(titles[i + 3])
            axes[1, i].set_xlabel('x [M]' if i == 0 else ('x [M]' if i == 1 else 'y [M]'))
            axes[1, i].set_ylabel('y [M]' if i == 0 else 'z [M]')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Emission')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['x-y slice', 'x-z slice', 'y-z slice']
        slices = [emission_3d[:, :, mid], emission_3d[:, mid, :],
                  emission_3d[mid, :, :]]
        for i in range(3):
            im = axes[i].imshow(slices[i].T, origin='lower', extent=extent,
                                cmap='afmhot', vmin=0)
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('x [M]' if i < 2 else 'y [M]')
            axes[i].set_ylabel('y [M]' if i == 0 else 'z [M]')
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Emission')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_movie_comparison(pred_movie, true_movie, t_frames, n_show=6,
                          save_path=None):
    """
    Show selected time frames: true vs predicted images.

    Parameters
    ----------
    pred_movie : np.ndarray, shape (n_frames, H, W)
    true_movie : np.ndarray, shape (n_frames, H, W)
    t_frames : np.ndarray, shape (n_frames,)
    n_show : int
    save_path : str, optional
    """
    indices = np.linspace(0, len(t_frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    vmax = max(true_movie.max(), pred_movie.max())

    for j, idx in enumerate(indices):
        axes[0, j].imshow(true_movie[idx], origin='lower', vmin=0,
                          vmax=vmax, cmap='afmhot')
        axes[0, j].set_title(f't={t_frames[idx]:.0f}M (true)')
        axes[0, j].axis('off')

        axes[1, j].imshow(pred_movie[idx], origin='lower', vmin=0,
                          vmax=vmax, cmap='afmhot')
        axes[1, j].set_title(f't={t_frames[idx]:.0f}M (pred)')
        axes[1, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_lightcurve(pred_images, true_images, t_frames, save_path=None):
    """
    Plot total flux (lightcurve) over time.

    Parameters
    ----------
    pred_images : np.ndarray, shape (n_frames, H, W)
    true_images : np.ndarray, shape (n_frames, H, W)
    t_frames : np.ndarray
    save_path : str, optional
    """
    pred_lc = pred_images.sum(axis=(-1, -2))
    true_lc = true_images.sum(axis=(-1, -2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_frames, true_lc, 'b-o', markersize=3, label='True')
    ax.plot(t_frames, pred_lc, 'r-x', markersize=4, label='Predicted')
    ax.set_xlabel('Time [M]')
    ax.set_ylabel('Total flux')
    ax.set_title('Lightcurve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_loss_curves(loss_history, save_path=None):
    """
    Plot training loss in log scale.

    Parameters
    ----------
    loss_history : list or np.ndarray
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3D Volume Rendering (ported from bhnerf/visualization.py VolumeVisualizer)
# ---------------------------------------------------------------------------

def _normalize(x):
    """Normalize a vector."""
    return x / (np.linalg.norm(x) + 1e-30)


def _viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def _generate_rays(camtoworlds, width, height, focal):
    """Generate camera rays (pinhole model)."""
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing='xy',
    )
    camera_dirs = np.stack([
        (x - width * 0.5 + 0.5) / focal,
        -(y - height * 0.5 + 0.5) / focal,
        -np.ones_like(x),
    ], axis=-1)
    directions = (camera_dirs[..., None, :] * camtoworlds[None, None, :3, :3]).sum(axis=-1)
    origins = np.broadcast_to(camtoworlds[None, None, :3, -1], directions.shape)
    return origins, directions


def render_volume_3d(emission_3d, domain_r, cam_r=None, azimuth=0.0,
                     zenith=np.pi / 3, resolution=128, n_samples=128,
                     bh_radius=2.0, cmap='hot', facewidth=None,
                     linewidth=0.1):
    """
    Render a 3D emission volume from a given camera angle.

    Faithfully ported from bhnerf's VolumeVisualizer + alpha_composite.

    Parameters
    ----------
    emission_3d : np.ndarray, shape (D, H, W)
        3D emission volume (regular grid from -domain_r to +domain_r).
    domain_r : float
        Half-width of the cubic domain (e.g., fov_M / 2).
    cam_r : float, optional
        Camera distance. Default: 3 * domain_r.
    azimuth, zenith : float
        Camera angles in radians.
    resolution : int
        Output image size (resolution x resolution).
    n_samples : int
        Number of ray-march samples.
    bh_radius : float
        Black hole radius (dark sphere).
    cmap : str
        Matplotlib colormap.
    facewidth : float, optional
        Wireframe cube width. Default: 2 * domain_r.
    linewidth : float
        Wireframe line thickness.

    Returns
    -------
    image : np.ndarray, shape (resolution, resolution, 3), float32
    """
    if cam_r is None:
        cam_r = 3.0 * domain_r
    if facewidth is None:
        facewidth = 2.0 * domain_r

    width, height = resolution, resolution
    up = np.array([0.0, 0.0, 1.0])

    # Camera setup (identical to original VolumeVisualizer.set_view)
    camorigin = cam_r * np.array([
        np.cos(azimuth) * np.sin(zenith),
        np.sin(azimuth) * np.sin(zenith),
        np.cos(zenith),
    ])
    vm = _viewmatrix(camorigin, up, camorigin)
    fov = 1.06 * np.arctan(np.sqrt(3) * domain_r / cam_r)
    focal = 0.5 * width / np.tan(fov)
    rays_o, rays_d = _generate_rays(vm, width, height, focal)

    near = cam_r - np.sqrt(3) * domain_r
    far = cam_r + np.sqrt(3) * domain_r
    t_vals = np.linspace(near, far, n_samples)
    pts = rays_o[..., None, :] + t_vals[None, None, :, None] * rays_d[..., None, :]
    # pts: (H, W, n_samples, 3)

    # Step distances (same as original self.d)
    dists = np.linalg.norm(
        np.concatenate([np.diff(pts, axis=2),
                        np.zeros_like(pts[:, :, -1:, :])], axis=2),
        axis=-1,
    )  # (H, W, n_samples)

    # --- Sample emission at ray points via trilinear interpolation ---
    D = emission_3d.shape[0]
    grid_coords = (pts + domain_r) / (2.0 * domain_r) * (D - 1)
    gi, gj, gk = grid_coords[..., 0], grid_coords[..., 1], grid_coords[..., 2]

    in_bounds = (
        (gi >= 0) & (gi <= D - 1) &
        (gj >= 0) & (gj <= D - 1) &
        (gk >= 0) & (gk <= D - 1)
    )
    gi_c = np.clip(gi, 0, D - 1)
    gj_c = np.clip(gj, 0, D - 1)
    gk_c = np.clip(gk, 0, D - 1)

    i0 = np.floor(gi_c).astype(int)
    j0 = np.floor(gj_c).astype(int)
    k0 = np.floor(gk_c).astype(int)
    i1 = np.minimum(i0 + 1, D - 1)
    j1 = np.minimum(j0 + 1, D - 1)
    k1 = np.minimum(k0 + 1, D - 1)
    di, dj, dk = gi_c - i0, gj_c - j0, gk_c - k0

    sampled = (
        emission_3d[i0, j0, k0] * (1 - di) * (1 - dj) * (1 - dk) +
        emission_3d[i0, j0, k1] * (1 - di) * (1 - dj) * dk +
        emission_3d[i0, j1, k0] * (1 - di) * dj * (1 - dk) +
        emission_3d[i0, j1, k1] * (1 - di) * dj * dk +
        emission_3d[i1, j0, k0] * di * (1 - dj) * (1 - dk) +
        emission_3d[i1, j0, k1] * di * (1 - dj) * dk +
        emission_3d[i1, j1, k0] * di * dj * (1 - dk) +
        emission_3d[i1, j1, k1] * di * dj * dk
    ) * in_bounds  # (H, W, n_samples)

    # --- Colormap & alpha (matching original render()) ---
    emax = sampled.max()
    if emax > 0:
        sampled_norm = sampled / emax
    else:
        sampled_norm = sampled

    cm = plt.get_cmap(cmap)
    emission_cm = cm(sampled_norm)                              # (H,W,S,4)
    emission_cm = np.clip(emission_cm - 0.05, 0.0, 1.0)
    # Replace alpha with normalised emission (original line 580)
    emission_cm[..., 3] = sampled_norm

    # --- Draw wireframe cube edges (fast vectorised version) ---
    hw = facewidth / 2.0
    # For each edge: points near two faces simultaneously form an edge
    edge_alpha = np.zeros_like(sampled)
    for axis_idx in range(3):
        other = [a for a in range(3) if a != axis_idx]
        for s0 in [-1, 1]:
            for s1 in [-1, 1]:
                da = np.abs(pts[..., other[0]] - s0 * hw)
                db = np.abs(pts[..., other[1]] - s1 * hw)
                d_edge = np.sqrt(da ** 2 + db ** 2)
                edge_alpha = np.maximum(edge_alpha, np.exp(-d_edge / linewidth ** 2))

    # Clamp edges to within cube
    outside_cube = np.amax(np.abs(pts), axis=-1) > (hw + 2 * linewidth)
    edge_alpha[outside_cube] = 0.0

    # Merge: edge colour is black, blended over emission
    line_rgba = np.zeros_like(emission_cm)
    line_rgba[..., 3] = edge_alpha
    emission_cm[..., :3] = (emission_cm[..., :3] * (1 - edge_alpha[..., None])
                            + 0.0 * edge_alpha[..., None])
    emission_cm[..., 3] = np.maximum(emission_cm[..., 3], edge_alpha)

    # --- Draw black hole sphere ---
    if bh_radius > 0:
        r_pts = np.linalg.norm(pts, axis=-1)
        bh_mask = r_pts < bh_radius
        lightdir = _normalize(np.array([-1.0, -1.0, 1.0]))
        bh_shade = np.sum(lightdir * pts, axis=-1)
        bh_shade = np.clip(bh_shade * 0.3, 0, 0.15)
        emission_cm[bh_mask, 0] = bh_shade[bh_mask]
        emission_cm[bh_mask, 1] = bh_shade[bh_mask]
        emission_cm[bh_mask, 2] = bh_shade[bh_mask]
        emission_cm[bh_mask, 3] = 1.0

    # Clip to [0, 1]
    emission_cm = np.clip(emission_cm, 0.0, 1.0)

    # --- Alpha compositing (matching original alpha_composite exactly) ---
    color = emission_cm[..., :3] * dists[0, :, :, None]  # scale by step dist
    alpha = emission_cm[..., 3:]                           # (H,W,S,1)

    inside_halfwidth = facewidth / 2.0 - linewidth
    inside = (np.amax(np.abs(pts), axis=-1) < inside_halfwidth).astype(np.float64)
    bh_outside = (np.linalg.norm(pts, axis=-1) > bh_radius).astype(np.float64)
    combined_mask = inside * bh_outside  # (H, W, S)

    rendering = np.zeros((height, width, 3), dtype=np.float64)
    acc = np.zeros((height, width), dtype=np.float64)

    for i in range(n_samples):
        ind = n_samples - 1 - i  # back to front

        # Inside cube & outside BH: direct emission accumulation (no alpha)
        rendering += combined_mask[:, :, ind, None] * color[:, :, ind, :]

        # Outside: standard alpha compositing
        outside_alpha = alpha[:, :, ind, :] * (1.0 - combined_mask[:, :, ind, None])
        rendering = rendering * (1.0 - outside_alpha) + color[:, :, ind, :] * outside_alpha

        acc = alpha[:, :, ind, 0] + (1.0 - alpha[:, :, ind, 0]) * acc

    # White background
    rendering += np.array([1.0, 1.0, 1.0])[None, None, :] * (1.0 - acc[:, :, None])
    rendering = np.clip(rendering, 0.0, 1.0)
    return rendering.astype(np.float32)


def plot_emission_3d(emission_3d, fov_M, n_views=6, bh_radius=2.0,
                     ground_truth=None, save_path=None):
    """
    Render the 3D emission volume from multiple camera angles.

    Produces a figure similar to bhnerf Tutorial 5: a row of volume-rendered
    images viewed from azimuth angles spanning 0 to 360 degrees.

    Parameters
    ----------
    emission_3d : np.ndarray, shape (D, H, W)
        3D emission volume.
    fov_M : float
        Field of view in M units.
    n_views : int
        Number of viewing angles.
    bh_radius : float
        Black hole radius for visualization.
    ground_truth : np.ndarray, optional
        If provided, show ground truth in top row and estimate in bottom row.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    domain_r = fov_M / 2.0
    azimuths = np.linspace(0, 2 * np.pi, n_views, endpoint=False)

    def _render_views(vol):
        imgs = []
        for az in azimuths:
            img = render_volume_3d(
                vol, domain_r, azimuth=az, zenith=np.pi / 3,
                resolution=128, n_samples=128,
                bh_radius=bh_radius, cmap='hot',
                facewidth=2.0 * domain_r, linewidth=0.15,
            )
            imgs.append(img)
        return imgs

    if ground_truth is not None:
        images_gt = _render_views(ground_truth)
        images_est = _render_views(emission_3d)
        fig, axes = plt.subplots(2, n_views, figsize=(2.8 * n_views, 5.6))
        for j in range(n_views):
            axes[0, j].imshow(images_gt[j])
            axes[0, j].set_title(f'{np.degrees(azimuths[j]):.0f}° (GT)')
            axes[0, j].axis('off')
            axes[1, j].imshow(images_est[j])
            axes[1, j].set_title(f'{np.degrees(azimuths[j]):.0f}° (Est)')
            axes[1, j].axis('off')
    else:
        images = _render_views(emission_3d)
        fig, axes = plt.subplots(1, n_views, figsize=(2.8 * n_views, 2.8))
        if n_views == 1:
            axes = [axes]
        for ax, img, az in zip(axes, images, azimuths):
            ax.imshow(img)
            ax.set_title(f'{np.degrees(az):.0f}°')
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def print_metrics_table(metrics):
    """
    Print metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
    """
    print("\n" + "=" * 50)
    print(f"{'Metric':<25} {'Value':>15}")
    print("=" * 50)
    for key, val in metrics.items():
        print(f"{key:<25} {val:>15.6f}")
    print("=" * 50)
