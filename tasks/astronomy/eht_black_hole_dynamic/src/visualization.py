"""
Visualization and metrics for dynamic black hole imaging.

Provides image quality metrics (NRMSE, NCC) and plotting utilities
for comparing ground truth and reconstructed video frames.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(est, ref):
    """Compute NRMSE and NCC between estimated and reference images.

    Args:
        est: (N, N) estimated image
        ref: (N, N) reference (ground truth) image

    Returns:
        dict with 'nrmse' and 'ncc'
    """
    est_flat = est.ravel()
    ref_flat = ref.ravel()

    # NRMSE: normalized root mean squared error
    nrmse = np.sqrt(np.mean((est_flat - ref_flat)**2)) / (ref_flat.max() - ref_flat.min() + 1e-30)

    # NCC: normalized cross-correlation
    est_c = est_flat - est_flat.mean()
    ref_c = ref_flat - ref_flat.mean()
    ncc_denom = np.sqrt(np.sum(est_c**2) * np.sum(ref_c**2))
    ncc = float(np.sum(est_c * ref_c) / (ncc_denom + 1e-30))

    return {'nrmse': float(nrmse), 'ncc': ncc}


def compute_video_metrics(est_frames, ref_frames):
    """Compute per-frame and average metrics for video reconstruction.

    Args:
        est_frames: list of (N, N) estimated frames
        ref_frames: list/array of (N, N) reference frames

    Returns:
        dict with 'per_frame' (list of dicts) and 'average' (dict)
    """
    per_frame = []
    for t in range(len(est_frames)):
        m = compute_metrics(est_frames[t], ref_frames[t])
        per_frame.append(m)

    avg_nrmse = np.mean([m['nrmse'] for m in per_frame])
    avg_ncc = np.mean([m['ncc'] for m in per_frame])

    return {
        'per_frame': per_frame,
        'average': {'nrmse': float(avg_nrmse), 'ncc': float(avg_ncc)},
    }


def print_metrics_table(metrics_dict):
    """Print a formatted comparison table of reconstruction metrics.

    Args:
        metrics_dict: dict mapping method_name -> video_metrics dict
    """
    print(f"\n{'Method':<30s}  {'Avg NRMSE':>10s}  {'Avg NCC':>10s}")
    print('-' * 56)
    for name, m in metrics_dict.items():
        avg = m['average']
        print(f"{name:<30s}  {avg['nrmse']:10.4f}  {avg['ncc']:10.4f}")
    print()


def unwrap_ring(frames, N, psize, ring_radius_frac=0.22, n_angle=360):
    """Unwrap intensity along a circle of fixed radius for each frame.

    For each frame, sample intensities around a circle centred on the image.
    Stack columns to form an (n_angle, n_frames) angle × time image.

    Args:
        frames: (n_frames, N, N) array or list of (N, N) images
        N: image size
        psize: pixel size in radians (unused, kept for API consistency)
        ring_radius_frac: radius of sampling circle as fraction of image
            half-width in pixel units (default 0.22 matches ground truth ring)
        n_angle: number of angular samples around the circle

    Returns:
        angle_time: (n_angle, n_frames) array of unwrapped intensities
        angles_deg: (n_angle,) array of angles in degrees
    """
    from scipy.ndimage import map_coordinates

    frames = np.asarray(frames)
    n_frames = frames.shape[0]
    angles_deg = np.linspace(0, 360, n_angle, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    # Circle centre and radius in pixel coordinates
    cy, cx = (N - 1) / 2.0, (N - 1) / 2.0
    r_pix = ring_radius_frac * (N / 2.0)

    angle_time = np.zeros((n_angle, n_frames))
    for t in range(n_frames):
        # Sample positions on the circle (x = cos, y = sin for origin='lower')
        sample_x = cx + r_pix * np.cos(angles_rad)
        sample_y = cy + r_pix * np.sin(angles_rad)
        coords = np.array([sample_y, sample_x])  # (row, col) for map_coordinates
        angle_time[:, t] = map_coordinates(frames[t], coords, order=3, mode='nearest')

    return angle_time, angles_deg


def plot_angle_time(gt_frames, recon_dict, frame_times, N, psize,
                    ring_radius_frac=0.22, save_path=None):
    """Plot angle × time diagrams (cf. Bouman et al. 2017, Fig. 14).

    Unwraps intensity around a fixed-radius circle for each frame and
    displays as an image with angle on the y-axis and time on the x-axis.
    A rotating hot spot appears as a diagonal stripe.

    Args:
        gt_frames: (n_frames, N, N) ground truth video
        recon_dict: dict mapping method_name -> list of (N, N) frames
        frame_times: (n_frames,) timestamps in hours
        N: image size
        psize: pixel size in radians
        ring_radius_frac: sampling circle radius as fraction of image half-width
        save_path: if given, save figure to this path
    """
    methods = {'Ground Truth': np.asarray(gt_frames)}
    methods.update(recon_dict)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(4.5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for i, (name, frames) in enumerate(methods.items()):
        at, angles = unwrap_ring(frames, N, psize, ring_radius_frac)
        ax = axes[i]
        ax.imshow(at, aspect='auto', cmap='afmhot', origin='lower',
                  extent=[frame_times[0], frame_times[-1], 0, 360],
                  interpolation='bilinear')
        ax.set_xlabel('Time (hours)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Angle (degrees)', fontsize=10)
        else:
            ax.set_yticks([])
        ax.set_title(name, fontsize=11)

    fig.suptitle('Angle × Time (intensity along ring)', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_video_comparison(gt_frames, recon_dict, frame_times=None,
                          save_path=None, n_display=6):
    """Plot ground truth vs reconstructed video frames side-by-side.

    Args:
        gt_frames: (n_frames, N, N) ground truth
        recon_dict: dict mapping method_name -> list of (N, N) frames
        frame_times: (n_frames,) timestamps (for title labels)
        save_path: if given, save figure to this path
        n_display: number of frames to display
    """
    n_frames = len(gt_frames)
    n_methods = len(recon_dict)
    step = max(1, n_frames // n_display)
    frame_indices = list(range(0, n_frames, step))[:n_display]

    fig, axes = plt.subplots(n_methods + 1, len(frame_indices),
                             figsize=(2.5 * len(frame_indices),
                                      2.5 * (n_methods + 1)))
    if len(frame_indices) == 1:
        axes = axes[:, np.newaxis]

    # Plot ground truth
    for j, idx in enumerate(frame_indices):
        ax = axes[0, j]
        ax.imshow(gt_frames[idx], cmap='afmhot', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if frame_times is not None:
            ax.set_title(f't={frame_times[idx]:.1f}h', fontsize=9)
        if j == 0:
            ax.set_ylabel('Ground Truth', fontsize=9)

    # Plot each method
    for row, (name, frames) in enumerate(recon_dict.items(), start=1):
        for j, idx in enumerate(frame_indices):
            ax = axes[row, j]
            ax.imshow(frames[idx], cmap='afmhot', origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)


def plot_metrics_over_time(metrics_dict, frame_times=None, save_path=None):
    """Plot per-frame NRMSE and NCC over time.

    Args:
        metrics_dict: dict mapping method_name -> video_metrics dict
        frame_times: (n_frames,) timestamps
        save_path: if given, save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for name, m in metrics_dict.items():
        pf = m['per_frame']
        nrmse = [f['nrmse'] for f in pf]
        ncc = [f['ncc'] for f in pf]
        x = frame_times if frame_times is not None else np.arange(len(pf))
        ax1.plot(x, nrmse, 'o-', label=name, markersize=4)
        ax2.plot(x, ncc, 'o-', label=name, markersize=4)

    ax1.set_xlabel('Time (hours)' if frame_times is not None else 'Frame')
    ax1.set_ylabel('NRMSE')
    ax1.set_title('NRMSE over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (hours)' if frame_times is not None else 'Frame')
    ax2.set_ylabel('NCC')
    ax2.set_title('NCC over Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
