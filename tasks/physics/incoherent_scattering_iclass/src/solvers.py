"""Local I-CLASS phase updates and paper-faithful output formation.

Extracted phase kernel: Weinberg/Sunray/Katz ICLASS.py. Complete MIT terms
and the independently validated output corrections: test/validation.md.
"""
import time
import torch
from .preprocessing import prepare_matrix
from .physics_model import phase_direction, estimate_mtf, rms_object, covariance_coherence

ITERATIONS = 250
SIGMA = 0.01


def solve(frames, iterations=ITERATIONS, sigma=SIGMA, device=None, record_steps=()):
    """Actively reconstruct all measurements; return arrays and diagnostics.

    Optional snapshots serve stage tests. Standard execution retains all
    150 illuminations and 160000 spatial Fourier rows. No reference is read.
    """
    if iterations < 1 or sigma <= 0:
        raise ValueError('Positive iterations and sigma required')
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(6)
    start = time.perf_counter()
    # Upstream forms the Fourier matrix on CPU, then transfers it to CUDA.
    matrix,uncorrected = prepare_matrix(frames, device='cpu')
    matrix = matrix.to(device)
    height,width = frames.shape[-2:]
    initial_coherence = covariance_coherence(matrix)
    mtf = estimate_mtf(matrix)
    phase = torch.ones(matrix.shape[0], dtype=torch.complex64, device=device)
    snapshots = {}
    for step in range(1,iterations+1):
        direction = phase_direction(matrix)
        correction = torch.exp(1j*direction.angle())
        phase *= correction
        matrix = (correction*matrix.T).T
        if step in record_steps:
            snapshots[step] = phase.detach().cpu().numpy().copy()
    final_coherence = covariance_coherence(matrix)
    phase_only = rms_object(matrix, (height,width))
    image = rms_object(matrix, (height,width), mtf, sigma)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start

    def public_map(vector):
        return vector.detach().cpu().numpy().reshape(width,height).T.copy()[None]

    result = {'reconstruction':image.detach().cpu().numpy()[None],
              'phase':public_map(phase), 'mtf':public_map(mtf)}
    details = {'runtime_seconds':elapsed, 'iterations':iterations, 'sigma':sigma,
               'device':str(device), 'initial_coherence':initial_coherence,
               'final_coherence':final_coherence,
               'phase_only':phase_only.detach().cpu().numpy()[None],
               'raw_rms':uncorrected.detach().cpu().numpy()[None],
               'snapshots':snapshots}
    return result,details
