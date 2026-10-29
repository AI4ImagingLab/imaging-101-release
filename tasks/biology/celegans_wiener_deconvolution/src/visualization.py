"""Review comparisons and independently defined scalar agreement metrics."""
import numpy as np

def agreement(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a, b = estimate.ravel().astype(float), reference.ravel().astype(float)
    return {'ncc': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            'nrmse': float(np.sqrt(np.mean((a-b)**2)) / np.ptp(b))}

def compare(observation: np.ndarray, upstream: np.ndarray, fresh: np.ndarray):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    low, high = np.percentile(observation, [1, 99.5])
    for ax, data, title in zip(axes[:3], [observation, upstream, fresh],
                               ['Repository Airyscan observation', 'Reproduced upstream Wiener', 'Fresh local Wiener']):
        ax.imshow(data, cmap='gray', vmin=low, vmax=high)
        ax.set_title(title, fontsize=9)
    im = axes[3].imshow(fresh-upstream, cmap='coolwarm')
    axes[3].set_title('Fresh minus upstream (a.u.)', fontsize=9)
    fig.colorbar(im, ax=axes[3], shrink=.65)
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    return fig
