from .physics_model import solve
def reconstruct(low_loss_counts,core_loss_counts):
    """Fresh Gaussian-resolved Fourier-ratio reconstruction and diagnostic stages."""
    return solve(low_loss_counts,core_loss_counts)
