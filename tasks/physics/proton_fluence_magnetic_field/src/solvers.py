"""Explicit log-parabolic Monge--Ampere relaxation, not a domain solver.

Adapted from PROBLEM, copyright (c)2017 Flash Center for Computational Science,
MIT License. Discretization and update order follow the independently run source.
"""
import time
import numpy as np
from scipy.fft import dctn, idctn
from .physics_model import transport_state, fluence_interpolator, magnetic_from_mapping
from .preprocessing import uniform_incident_fluence

DT = 1e-8
TOL = 1e-3
MAX_STEPS = 400000

def reconstruct(x, y, fluence, dt=DT, tol=TOL, max_steps=MAX_STEPS,
                snapshot_steps=(0, 1000, 10000, 50000, 100000, 200000, 399999)):
    """Fresh full selected inverse; return field, source-order state and history.

    Stop after updating Phi if the pre-update residual 2-norm is below tol,
    or at max_steps. The returned map is the LAST PRE-UPDATE gradient, retaining
    the original source convention. No hidden reference, checkpoint or cache read.
    """
    start = time.perf_counter()
    incident = uniform_incident_fluence(fluence)
    interpolate = fluence_interpolator(x, y, fluence)
    potential = .5 * (x ** 2 + y ** 2)
    rms_update = np.zeros(max_steps)
    norms = np.zeros(max_steps)
    snapshots = {}
    negative_determinants = 0
    for k in range(max_steps):
        if not np.isfinite(potential).all():
            raise FloatingPointError('Non-finite mapping potential')
        state = transport_state(potential, x, y, incident, interpolate)
        update = potential + dt * state['Fn']
        rms_update[k] = np.sqrt(np.mean((update - potential) ** 2))
        norms[k] = np.linalg.norm(state['Fn'])
        negative_determinants += int(np.count_nonzero(state['det'] <= 0))
        potential = update
        if k in snapshot_steps:
            snapshots[k] = magnetic_from_mapping(x, y, state['phix'], state['phiy'])
        if norms[k] < tol:
            break
    field = magnetic_from_mapping(x, y, state['phix'], state['phiy'])
    return dict(reconstruction=field, phin=potential, **state,
                RMSfluxerror=rms_update, residual_norm=norms[:k + 1],
                snapshots=snapshots, iterations=k + 1,
                stopping='tolerance' if norms[k] < tol else 'maximum_steps',
                negative_determinant_count=negative_determinants,
                runtime_seconds=time.perf_counter() - start)

def linear_poisson(x, y, fluence):
    """Computed weak-contrast comparison, not the selected nonlinear method.

    Linearize f(grad Phi)*det(Hess Phi)=mean(f) about the identity, then solve
    Laplacian u = 1-f/mean(f) with cell-centered homogeneous Neumann stencil.
    A DCT-II diagonalizes that stated discrete Laplacian; zero mean fixes gauge.
    """
    dx, dy = x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]
    nx, ny = fluence.shape
    rhs = 1 - fluence / np.mean(fluence)
    eigx = -4 * np.sin(np.pi * np.arange(nx) / (2 * nx)) ** 2 / dx ** 2
    eigy = -4 * np.sin(np.pi * np.arange(ny) / (2 * ny)) ** 2 / dy ** 2
    eigenvalues = eigx[:, None] + eigy[None, :]
    spectrum = dctn(rhs, type=2, norm='ortho')
    eigenvalues[0, 0] = 1
    spectrum /= eigenvalues; spectrum[0, 0] = 0
    perturbation = idctn(spectrum, type=2, norm='ortho')
    ux, uy = np.gradient(perturbation, dx, dy)
    ux[0, :] = 0; ux[-1, :] = 0; uy[:, 0] = 0; uy[:, -1] = 0
    return magnetic_from_mapping(x, y, x + ux, y + uy)
